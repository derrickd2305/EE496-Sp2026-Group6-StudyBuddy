"""
EE496 - group 6
Study Buddy :)
host-side protocol bridge to communicate in colon delimiited ASCII between the laptop and the Teensy v4.1
"""

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import serial
import serial.tools.list_ports

# constants ----------------------------------------------------------------------

SLEEPY_WAV  = "PERCLOSCHIME.WAV"    # sleepy chime file name (repeats at set interval)
POSTURE_WAV = "POSTURECHIME.WAV"    # posture chime file name (will play on rising edge of bad posture)
SLEEPY_REPEAT_SECS = 3.0            # interval between sleepy chimes while sleepy
STATUS_FLUSH_HZ = 5                 # rate to push updates to teensy
PING_INTERVAL = 1.0                 # seconds between pings if otherwise idle

@dataclass
class SBState:
    connected: bool = False
    fw_alarm: bool = False               # last STATE alarm flag from teensy
    fw_volume: int = 50                  # last STATE volume
    fw_wav_playing: bool = False
    last_event: Optional[str] = None
    last_ack: Optional[str] = None
    last_log: Optional[str] = None


class SerialBridge:
    def __init__(
            self,
            port: Optional[str] = None,
            baud: int = 115200,
            on_log: Optional[Callable[[str, str], None]] = None,
            on_ack: Optional[Callable[[str], None]] = None,
            on_state: Optional[Callable[[dict], None]] = None,
        ):
        self.port = port or self._autodetect()
        self.baud = baud
        self._ser: Optional[serial.Serial] = None

        # outbound: high-priority queue (alerts, calibrate), LCD updates are coalesced (only latest of x frames)
        # gets sent at STATUS_FLUSH_HZ.
        self._tx_queue: queue.Queue = queue.Queue(maxsize=128)
        self._latest_lcd: dict = {}             # keyed by 2ndary-verb (STATUS/PERCLOS/POSTURE/MSG)
        self._latest_lock = threading.Lock()

        # sleepy looping state
        self._sleepy_active = False
        self._sleepy_last_send = 0.0

        self._stop = threading.Event()
        self._rx_thread: Optional[threading.Thread] = None
        self._tx_thread: Optional[threading.Thread] = None

        self.state = SBState()
        self._on_log   = on_log   or (lambda msg, lvl: print(f"[TEENSY/{lvl}] {msg}"))
        self._on_ack   = on_ack   or (lambda verb: None)
        self._on_state = on_state or (lambda st:   None)

    # --------- lifecycle ----------------------------------------------------

    def start(self):
        if self.port is None:
            print("[BRIDGE] no serial port found, running in disconnected mode")
            return
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.1)
            self.state.connected = True
            print(f"[BRIDGE] opened {self.port} @ {self.baud}")
        except Exception as e:
            print(f"[BRIDGE] failed to open {self.port}: {e}")
            self._ser = None
            return
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._tx_thread = threading.Thread(target=self._tx_loop, daemon=True)
        self._rx_thread.start()
        self._tx_thread.start()

    def stop(self):
        self._stop.set()
        if self._ser is not None:
            try:
                if self._ser.is_open:                  
                    self._ser.close()
            except Exception:
                pass
        self.state.connected = False

    # --------- public API------------------

    # alerts: sleepy vs posture, true vs false
    
    
    # posture on -> fire posture wav once (firmware-side cooldown not used; host already debounces)
    def send_alert(self, kind: str, on: bool, reason: Optional[str] = None):
        if kind == "sleepy":
            # sleepy on -> start looping the perclos wav + tone alarm
            if on:
                self._sleepy_active = True
                self._sleepy_last_send = 0.0      # first tone is forced
                self._enqueue(f"WAV:{SLEEPY_WAV}")
            # sleepy off -> stop looping (and silence tone alarm)
            else:
                self._sleepy_active = False     # stop loop
                self._enqueue("WAV_STOP")       # interrupt any ongoing wav
                self._enqueue("ALARM:OFF")      # *might* not be necessary idk
        elif kind == "posture":
            # posture on -> only one wav
            if on:
                self._on_log(f"posture chime trigger: {reason}", "info")
                self._enqueue(f"WAV:{POSTURE_WAV}")
                if reason: 
                    self._set_lcd("MSG", f"posture: {reason}"[:28])    # # show the reason on the LCD bottom row (truncated to 28 letters, maybe not necessary)
            # posture "off" -> nothing to do

        # fallback
        else:
            self._on_log(f"unknown alert kind: {kind}", "warn")

    # only the latest values get sent at STATUS_FLUSH_HZ (merges multiple frames)
    def send_status(self, perclos: float, posture: float, state: str,
                    ear: Optional[float] = None):
        self._set_lcd("STATUS",  state)
        self._set_lcd("PERCLOS", int(round(max(0.0, min(1.0, perclos)) * 100)))
        self._set_lcd("POSTURE", int(round(max(0.0, min(1.0, posture)) * 100)))

    # called every callibration event
    def send_calibrate(self):
        self._set_lcd("STATUS", "calibrating")
        self._enqueue("LCD:MSG:recalibrating...")

    # motor commands
    def send_cmd_yaw(self, direction: str, angle: Optional[int] = None):
        # current model doesn't have angles (didn't use IMU)
        if angle is not None:
            self._enqueue(f"YAW:{int(angle)}")
            return
        # current model just makes small adjustments, 10 degrees at a time
        deltas = {"left": -10, "right": 10, "center": 0}
        d = deltas.get(direction)
        if d is None:
            return
        self._enqueue(f"YAW_REL:{d}" if d != 0 else "YAW:90")   # default 90 degrees, "straight"

    def send_cmd_height(self, direction: str):
        if direction == "up":
            self._enqueue("HEIGHT_MS:200")
        elif direction == "down":
            self._enqueue("HEIGHT_MS:-200")
        elif direction == "stop":
            self._enqueue("HEIGHT_STOP")

    def send_mode(self, speaker: str):
        pass

    # --------- USE THESE FUNCTIONS IF YOU NEED TO DEBUG (OR USE SERIAL MONITOR (EASIER)) -------------------------------------

    def send_volume(self, percent: int):
        p = max(0, min(100, int(percent)))
        self._enqueue(f"VOL:{p}")

    def send_wav(self, filename: str):
        self._enqueue(f"WAV:{filename}")

    def send_alarm(self, on: bool):
        self._enqueue("ALARM:ON" if on else "ALARM:OFF")

    def send_chime(self):
        self._enqueue("CHIME")

    def send_raw(self, line: str):
        self._enqueue(line)

    # --------- internal -----------------------------------------------------

    def _set_lcd(self, sub: str, value):
        with self._latest_lock:
            self._latest_lcd[sub] = value

    def _enqueue(self, line: str):
        try:
            self._tx_queue.put_nowait(line)
        # if the queue is full just drop the oldest command so we don't block main loop
        except queue.Full:
            try:
                self._tx_queue.get_nowait()
                self._tx_queue.put_nowait(line)
            except queue.Empty:
                pass

    def _autodetect(self) -> Optional[str]:
        ports = list(serial.tools.list_ports.comports())
        # PJRC vendor ID first
        for p in ports:
            if p.vid == 0x16C0:
                return p.device
        for p in ports:
            if "Teensy" in (p.description or ""):
                return p.device
        return ports[0].device if ports else None

    def _tx_loop(self):
        last_ping = 0.0
        last_status_flush = 0.0
        flush_interval = 1.0 / STATUS_FLUSH_HZ

        while not self._stop.is_set():
            now = time.time()
            pending: list[str] = []

            # 1. flush coalesced LCD updates at STATUS_FLUSH_HZ
            if now - last_status_flush >= flush_interval:
                last_status_flush = now
                with self._latest_lock:
                    snapshot = dict(self._latest_lcd)
                    self._latest_lcd.clear()
                if "STATUS" in snapshot:
                    pending.append(f"LCD:STATUS:{snapshot.pop('STATUS')}")
                for sub, val in snapshot.items():
                    pending.append(f"LCD:{sub}:{val}")

            # 2. drain the high-priority queue
            try:
                while True:
                    pending.append(self._tx_queue.get_nowait())
            except queue.Empty:
                pass

            # 3. re-trigger sleepy wav while still sleepy
            if self._sleepy_active and (now - self._sleepy_last_send) >= SLEEPY_REPEAT_SECS:
                self._sleepy_last_send = now
                pending.append(f"WAV:{SLEEPY_WAV}")

            # 4. heartbeat ping if otherwise idle
            if not pending and (now - last_ping) > PING_INTERVAL:
                last_ping = now
                pending.append("PING")

            # 5. write everything
            if self._ser is not None:
                for line in pending:
                    try:
                        self._ser.write((line + "\n").encode("ascii", errors="replace"))
                    except Exception as e:
                        self._on_log(f"tx error: {e}", "error")
                        self.state.connected = False
                        return

            time.sleep(0.01)        # ~100 Hz tick

    def _rx_loop(self):
        if self._ser is None:
            return
        buf = b""
        while not self._stop.is_set():
            try:
                data = self._ser.read(256)
            except Exception as e:
                self._on_log(f"rx error: {e}", "error")
                self.state.connected = False
                return
            if not data:
                continue
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip().decode("ascii", errors="replace")
                if not line:
                    continue
                self._dispatch(line)

    def _dispatch(self, line: str):
        # protocol replies are colon-delimited: BOOT, PONG, ACK:verb,
        # ERR:reason, LOG:text, STATE:k=v,k=v
        if line == "BOOT":
            self._on_log("teensy booted", "info")
            return
        if line == "PONG":
            return

        if line.startswith("ACK:"):
            verb = line[4:].strip()
            self.state.last_ack = verb
            self._on_ack(verb)
            return

        if line.startswith("ERR:"):
            self._on_log(f"firmware error: {line[4:]}", "error")
            return

        if line.startswith("LOG:"):
            text = line[4:].strip()
            self.state.last_log = text
            self._on_log(text, "info")
            return

        if line.startswith("STATE:"):
            # STATE:ALARM=0,VOL=50,WAV=0
            parts = line[6:].split(",")
            kv = {}
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    kv[k.strip()] = v.strip()
            try:
                if "ALARM" in kv:
                    self.state.fw_alarm = (kv["ALARM"] == "1")
                if "VOL" in kv:
                    self.state.fw_volume = int(kv["VOL"])
                if "WAV" in kv:
                    self.state.fw_wav_playing = (kv["WAV"] == "1")
            except ValueError:
                pass
            self._on_state(kv)
            return

        # unknown message - log it but don't crash
        self._on_log(f"unparsed: {line}", "warn")
