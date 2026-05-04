"""
USC EE496 CAPSTONE PROJECT 
STUDY BUDDY :)
GOALS: CAPTURE WEBCAM FOOTAGE, MEASURE PERCLOS/SHOULDER TILT/EAR TILT/FACE DISTANCE in order to assess the user's awakeness and posture while studying 
  _ _
 ( _ )
( (=) )
 (_ _)
   |
   |
   ' 
"""

import cv2
import sys
import time
import threading
import winsound
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks import python as mp_tasks
import urllib.request
import os

from serial_comm import SerialBridge

# _______________________________________________
# CONFIGURATION
# _______________________________________________

# for webcam access
CAMERA_INDEX = 1        # if 0 doesn't work, change this number to 1,2,... (should only be necessary if u have >1 camera)
WINDOW_NAME = "webcam | press Q to quit or press S to screenshot"
INFERENCE_WIDTH = 640   # frame is downscaled to this width before running MediaPipe (preview stays at full res)
INFERENCE_HEIGHT = 360  # ~half of 1280x720, keeps 16:9 aspect ratio

# for perclos
EAR_THRESHOLD = 0.15        # if EAR <= EAR_THRESHOLD: eyes == closed (was 0.10, too low to catch drowsy partial closure)
EAR_OPEN_THRESHOLD = 0.18   # EAR must grow above this value to count as reopened (tight hysteresis band)
BLINK_MAX_SECS = 0.4        # eye closures shorter than this are treated as blinks and don't count toward PERCLOS
PERCLOS_WINDOW = 60         # size of window, in seconds
SLEEPY_THRESHOLD = 0.20     # if perclos >= SLEEPY_THRESHOLD: trigger alarm
WAKE_WINDOW_SECS = 1.5      # seconds looked at this many seconds of recent data to see if user has woken up after an alarm
WAKE_OPEN_RATIO = 0.75      # PERCLOS ratio needed within WAKE_WINDOW_SECS after an alarm to detect "reawaken" status
WAKE_TRUNCATE_SECS = 5      # when reawoken, prune perclos data to this many seconds
WAKE_MIN_ALARM_SECS = 1.0   # alarm must have been on for at least this long before wake detection can fire

# for posture
POSTURE_CALIBRATION_SECS = 5        # time the user must sit straight upon boot up and every calibration
FORWARD_HEAD_THRESHOLD = 0.15       # ratio of deviation from baseline
SHOULDER_TILT_THRESHOLD = 8.0       # degress of deviation from baseline
POSTURE_WINDOW = 10                 # size of window, in seconds
BAD_POSTURE_THRESHOLD = 0.40        # if % of frames posture is bad -> alert user
HEAD_TILT_THRESHOLD = 15.0          # degrees of head tilt deviation from baseline
FACE_CLOSE_RATIO = 1.40             # face width >= baseline * this ratio -> user is leaning in / too close (bad posture)
POSTURE_ALARM_COOLDOWN = 15.0       # seconds between posture chimes so user isn't spammed
POSTURE_ALARM_FREQ = 600            # Hz - lower than sleepy alarm (880) so user can tell them apart
POSTURE_ALARM_DURATION_MS = 250     # ms - single chime duration

# for motor signals
POSE_CONFIDENCE_LOW = 0.55          # below this -> landmark confidence low
FACE_CONFIDENCE_LOW = 0.55          # below this -> face hard to see
SHOULDER_ASYMMETRY_MOTOR = 15.0     # camera is off center by 'x' degrees -> adjust yaw
TOO_CLOSE_SHOULDER_VIS = 0.5        # if cam is centered but shoulder vis is below this -> too close
TOO_CLOSE_SHOULDER_Y = 0.92         # if shoulders are below this y -> too close
TOO_CLOSE_NOSE_CENTER_LOW = 0.3     # LOW < nose x < HIGH -> centered
TOO_CLOSE_NOSE_CENTER_HIGH = 0.7    # LOW < nose x < HIGH -> centered
TOO_CLOSE_FACE_WIDTH_RATIO = 0.35   # face width/ frame width (used to differentiate between too_close and move camera up/down)
CAMERA_LOW_FACE_Y = 0.30            # face center y below this -> camera is too low
CAMERA_HIGH_FACE_Y = 0.65           # face center y above this -> camera is too high

# ____________________________________________________
# MEDIAPIPE SETUP
# ____________________________________________________

# face landmarking ----------

# mediapipe==0.10.30
FACE_MODEL_PATH = "face_landmarker.task"
FACE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
 
if not os.path.exists(FACE_MODEL_PATH):
    urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
 
face_options = FaceLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face_landmarker = FaceLandmarker.create_from_options(face_options)

# pose landmarking ------------------

POSE_MODEL_PATH = "pose_landmarker_lite.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

if not os.path.exists(POSE_MODEL_PATH):
    urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)

pose_options = PoseLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
pose_landmarker = PoseLandmarker.create_from_options(pose_options)

#landmark indices for face
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

#landmark indices for pose (only including upper body ones) [[maybe change later]]
NOSE = 0
LEFT_EAR_POSE = 7       # measuring head titl by comparing angle between ears / shoulders
RIGHT_EAR_POSE = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


#________________________________________________________
# EAR CALCULATION
# _______________________________________________________

# note: eye aspect ratio (EAR) = vertical distance / horizontal distance

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    points = []
    for i in eye_indices:
        lm = landmarks[i]
        points.append((lm.x * w, lm.y * h))
    
    v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    h1 = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    
    ear = (v1 + v2) / (2.0 * h1 + 1e-6)     # returns a float
    return ear

# ___________________________________________________
# POSTURE CALCULATION
# ___________________________________________________

# how far is the nose from the midpoint between shoulders | (+) value -> leaning towards cam, (-) value -> leaning away, 0 -> good
def forward_head_ratio(landmarks, w, h):
    nose = landmarks[NOSE]
    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]

    shoulder_mid_x = (ls.x + rs.x) / 2.0
    shoulder_width = abs(ls.x - rs.x)

    offset = (nose.x - shoulder_mid_x) / (shoulder_width + 1e-6)

    return offset

# angle between shoulder line and horizontal | 0 -> level, (+) -> left shoulder higher, (-) -> right shoulder higher
def shoulder_tilt_deg(landmarks, w, h):
    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]
    dy = (ls.y - rs.y) * h
    dx = (ls.x - rs.x) * w

    return np.degrees(np.arctan2(dy,dx))

# angle between ear line and horizontal | 0 -> level, (+) -> leaning right, (-) -> leaning left
def head_tilt_deg(landmarks, w, h):
    le = landmarks[LEFT_EAR_POSE]
    re = landmarks[RIGHT_EAR_POSE]
    dy = (le.y - re.y) * h
    dx = (le.x - re.x) * w
    return np.degrees(np.arctan2(dy, dx))

# average visibility of upper body landmarks, [0,1]
def pose_visibility_score(landmarks):
    upper_indices = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER]
    scores = [landmarks[i].visibility for i in upper_indices]

    return float(np.mean(scores))

# used to determine closeness to camera and the edges of the frame
def face_metrics(face_lms, w, h):
    xs = [lm.x for lm in face_lms]
    ys = [lm.y for lm in face_lms]
    face_width = max(xs) - min(xs)
    face_center_y = (min(ys) + max(ys))/2.0
    return face_width, face_center_y

#___________________________________________________
# MOTOR REPOSITIONING
#___________________________________________________

class MotorSignal:

    DEBOUNCE_FRAMES = 30        # window for posture persisting
    HOLD_SECS = 3.0             # time a signla stays visible in overlay after triggering, in seconds
    MOTOR_REPEAT_SECS = 0.5     # re-fire each motor cmd this often while its condition holds

    def __init__(self, bridge=None):
        self.bridge = bridge

        self._counters = {
            "yaw_left": 0,
            "yaw_right": 0,
            "height_up": 0,
            "height_down": 0,
            "too_close": 0,
        }

        self.active_signals = []    # list that contains strings of commands
        self._signal_times = {}

        
        self._sent_cmds = set()     # track which motor commands we've already sent so we don't spam the teensy
        self._last_sent_at = {}     # tracks when each command was sent, used to refire signals periodically

    # called once per frame to update internal counters and active signals
    def update(self, pose_lms, face_lms, w, h):
        now = time.time()
        triggered = set()                       # signals fired this frame
        fired_cmds = set()                      # motor commands fired this frame (subset of triggered)
        face_detected = face_lms is not None

        # can't see body at all
        if pose_lms is None:
            self._increment("height_up")
            self._reset("height_down")
            self._reset("yaw_left")
            self._reset("yaw_right")
            self._reset("too_close")
            if self._counters["height_up"] >= self.DEBOUNCE_FRAMES:
                triggered.add("MOTOR: raise camera height")
                fired_cmds.add("height_up")
        else:
            tilt = shoulder_tilt_deg(pose_lms, w, h)
            nose_x_norm = pose_lms[NOSE].x

            # yaw adjustment based on shoulder positions
            if tilt > SHOULDER_ASYMMETRY_MOTOR:
                self._increment("yaw_right")
                self._reset("yaw_left")
                if self._counters["yaw_right"] >= self.DEBOUNCE_FRAMES:
                    triggered.add("MOTOR: rotate camera right (off-center)")
                    fired_cmds.add("yaw_right")
            elif tilt < -SHOULDER_ASYMMETRY_MOTOR:
                self._increment("yaw_left")
                self._reset("yaw_right")
                if self._counters["yaw_left"] >= self.DEBOUNCE_FRAMES:
                    triggered.add("MOTOR: rotate camera left (off-center)")
                    fired_cmds.add("yaw_left")
            else:
                self._reset("yaw_left")
                self._reset("yaw_right")

            # yaw adjustment from edge of frame
            if nose_x_norm < 0.25:
                triggered.add("MOTOR: rotate camera left (user near edge)")
                fired_cmds.add("yaw_left")
            elif nose_x_norm > 0.75:
                triggered.add("MOTOR: rotate camera right (user near edge)")
                fired_cmds.add("yaw_right")

            # height adjustment if pose is fine and visible but face isn't visible
            if not face_detected:
                triggered.add("MOTOR: raise camera (face not visible)")
                fired_cmds.add("height_up")

            # "shoulders missing" cases: --> pick between too_close / height_up
            ls = pose_lms[LEFT_SHOULDER]
            rs = pose_lms[RIGHT_SHOULDER]
            shoulder_vis = min(ls.visibility, rs.visibility)
            shoulders_near_bottom = (ls.y > TOO_CLOSE_SHOULDER_Y) or (rs.y > TOO_CLOSE_SHOULDER_Y)
            face_centered = face_detected and (TOO_CLOSE_NOSE_CENTER_LOW < nose_x_norm < TOO_CLOSE_NOSE_CENTER_HIGH)
            shoulders_not_visible = face_centered and (shoulder_vis < TOO_CLOSE_SHOULDER_VIS or shoulders_near_bottom)

            if shoulders_not_visible and face_detected:
                fw, fcy = face_metrics(face_lms, w, h)

                # face is large -> too close to camera
                if fw > TOO_CLOSE_FACE_WIDTH_RATIO:
                    self._increment("too_close")
                    self._reset("height_up")
                    self._reset("height_down")
                    if self._counters["too_close"] >= self.DEBOUNCE_FRAMES:
                        triggered.add("USER: move camera back (too close)")
                # face sits high in frame -> camera too low
                elif fcy < CAMERA_LOW_FACE_Y:
                    self._increment("height_up")
                    self._reset("too_close")
                    self._reset("height_down")
                    if self._counters["height_up"] >= self.DEBOUNCE_FRAMES:
                        triggered.add("MOTOR: raise camera height")
                        fired_cmds.add("height_up")
                # face sits low in frame -> camera too high
                elif fcy > CAMERA_HIGH_FACE_Y:
                    self._increment("height_down")
                    self._reset("too_close")
                    self._reset("height_up")
                    if self._counters["height_down"] >= self.DEBOUNCE_FRAMES:
                        triggered.add("MOTOR: lower camera height")
                        fired_cmds.add("height_down")
                # face centered, but shoulders missing -> too closee
                else:
                    self._increment("too_close")
                    if self._counters["too_close"] >= self.DEBOUNCE_FRAMES:
                        triggered.add("USER: move camera back (too close)")
            else:
                # shoulders visible -> nothing to disambiguate; clear all three counters
                self._reset("too_close")
                self._reset("height_up")
                self._reset("height_down")

        # only send motor commands if either it is the rising edge, or if the signal has persisted for more than MOTOR_REPEAT_SECS
        if self.bridge is not None:
            for cmd in fired_cmds:
                last = self._last_sent_at.get(cmd, 0)
                rising_edge = cmd not in self._sent_cmds
                periodic    = (now - last) >= self.MOTOR_REPEAT_SECS
                if not (rising_edge or periodic):
                    continue
                if cmd == "yaw_left":
                    self.bridge.send_cmd_yaw("left")
                elif cmd == "yaw_right":
                    self.bridge.send_cmd_yaw("right")
                elif cmd == "height_up":
                    self.bridge.send_cmd_height("up")
                elif cmd == "height_down":
                    self.bridge.send_cmd_height("down")
                self._last_sent_at[cmd] = now
        self._sent_cmds = fired_cmds

        # update hold timestamps for anything triggered this frame
        for sig in triggered:
            self._signal_times[sig] = now

        # active_signals = anything still within its hold window
        self.active_signals = [
            sig for sig, t in self._signal_times.items()
            if now - t <= self.HOLD_SECS
        ]
 
    def _increment(self, key):
        self._counters[key] = min(self._counters[key] + 1, self.DEBOUNCE_FRAMES + 1)
 
    def _reset(self, key):
        self._counters[key] = 0

# __________________________________________________
# SOUND ALERT (plays on laptop using windows sound library, for debugging purposes) (always "on", just turned off by muting laptop sound)
# __________________________________________________

alarm_active = False

def sound_alarm():
    global alarm_active
    while alarm_active:
        if not alarm_active:
            break
        winsound.Beep(880, 150)   # frequency Hz, duration ms
        for _ in range(6):
            if not alarm_active:
                return
            time.sleep(0.5)
 
def start_alarm():
    global alarm_active
    if not alarm_active:
        alarm_active = True
        threading.Thread(target=sound_alarm, daemon=True).start()
 
def stop_alarm():
    global alarm_active
    alarm_active = False

# posture chime: fires once per event 
_posture_chime_playing = False
_last_posture_chime_time = 0.0

def _play_posture_chime():
    global _posture_chime_playing
    try:
        # two beeps for posture (one beep for perclos)
        winsound.Beep(POSTURE_ALARM_FREQ, POSTURE_ALARM_DURATION_MS)
        winsound.Beep(int(POSTURE_ALARM_FREQ * 0.75), POSTURE_ALARM_DURATION_MS)
    finally:
        _posture_chime_playing = False

def trigger_posture_chime():
    global _posture_chime_playing, _last_posture_chime_time
    now = time.time()
    if _posture_chime_playing:
        return
    if now - _last_posture_chime_time < POSTURE_ALARM_COOLDOWN:
        return
    _posture_chime_playing = True
    _last_posture_chime_time = now
    threading.Thread(target=_play_posture_chime, daemon=True).start()

# ______________________________________________________
# PERCLOS TRACKER
# ______________________________________________________

class PercloseTracker:

    def __init__(self, window_seconds=PERCLOS_WINDOW, blink_max_secs=BLINK_MAX_SECS):
        self.window = window_seconds
        self.blink_max_secs = blink_max_secs

        self.closures = []                      # completed closure intervals that exceeded the blink filter, each element: (start, end)
        self._current_close_start = None        # state for the currently-in-progress closure
        self._first_seen = None                 # timestamp for beginning of closure
        self._last_seen = None                  # timestamp for end of closure

    def update(self, is_closed: bool):
        now = time.time()

        if self._first_seen is None:
            self._first_seen = now
        self._last_seen = now

        if is_closed:
            # start of a new closure
            if self._current_close_start is None:
                self._current_close_start = now
        else:
            # closure just ended
            if self._current_close_start is not None:
                duration = now - self._current_close_start
                # only commit it if it was long enough to not be a blink
                if duration >= self.blink_max_secs:
                    self.closures.append((self._current_close_start, now))
                self._current_close_start = None

        self._prune()

    # drop closures that have fully scrolled off the window
    def _prune(self):
        if self._last_seen is None:
            return
        cutoff = self._last_seen - self.window
        self.closures = [(s, e) for s, e in self.closures if e >= cutoff]
        # denominator <= self.window
        if self._first_seen is not None and self._first_seen < cutoff:
            self._first_seen = cutoff

    def score(self):
        if self._last_seen is None or self._first_seen is None:
            return 0.0

        observed = self._last_seen - self._first_seen
        if observed <= 0:
            return 0.0

        cutoff = self._last_seen - self.window

        # sum the time spent in qualifying closures, clipped to the window
        closed_time = 0.0
        for s, e in self.closures:
            s_clipped = max(s, cutoff)
            e_clipped = min(e, self._last_seen)
            if e_clipped > s_clipped:
                closed_time += e_clipped - s_clipped

        # include an in-progress closure only if it has already crossed the blink threshold (no more jumping %)
        if self._current_close_start is not None:
            in_progress = self._last_seen - self._current_close_start
            if in_progress >= self.blink_max_secs:
                s_clipped = max(self._current_close_start, cutoff)
                closed_time += self._last_seen - s_clipped

        return closed_time / observed

    def seconds_of_data(self):
        if self._first_seen is None or self._last_seen is None:
            return 0.0
        return self._last_seen - self._first_seen

    # used for wake detection, fraction of the last frames that eyes were OPEN
    def recent_open_ratio(self, secs=1.5):
        if self._last_seen is None:
            return 0.0
        window_start = self._last_seen - secs

        closed_time = 0.0
        for s, e in self.closures:
            s_clipped = max(s, window_start)
            e_clipped = min(e, self._last_seen)
            if e_clipped > s_clipped:
                closed_time += e_clipped - s_clipped

        if self._current_close_start is not None:
            in_progress = self._last_seen - self._current_close_start
            if in_progress >= self.blink_max_secs:
                s_clipped = max(self._current_close_start, window_start)
                closed_time += self._last_seen - s_clipped

        observed = min(secs, self._last_seen - self._first_seen) if self._first_seen is not None else 0
        if observed <= 0:
            return 0.0
        return 1.0 - (closed_time / observed)

    # used when a clear wake event is detected
    def reset(self):
        self.closures = []
        self._current_close_start = None
        self._first_seen = None
        self._last_seen = None
    
#________________________________________________________
# POSTURE TRACKING
#________________________________________________________

class PostureTracker:
    def __init__(self):
        self.window = POSTURE_WINDOW
        self.history = []               # (timestamp, is_bad)

        self.calibrating = True         # calibration "state"
        self.cal_samples = []           # (fh_ratio, sh_tilt, head_tilt, face_width)
        self.cal_start = None
        self.baseline_fh = None         # calibrated forward head ratio
        self.baseline_tilt = None       # calibrated shoulder tilt
        self.baseline_head_tilt = None  # calibrated head tilt
        self.baseline_face_width = None # calibrated face width (proxy for distance to camera)

        # track the last reason so the overlay / alarm can show WHY posture was bad
        self.last_reason = None
    
    # call to begin a new calibration (upon boot or after motor movements)
    def start_calibration(self):
        self.calibrating = True
        self.cal_samples = []
        self.cal_start = time.time()
        self.history = []
        self.last_reason = None
        print("[POSTURE] calibration has started, please sit upright")

    # once per frame, returns a string used to track state in main [no_pose, calibrating, bad, ok] 
    def update(self, pose_lms, face_lms, w, h):
        if pose_lms is None:
            return "no_pose"        
        
        fh = forward_head_ratio(pose_lms, w, h)
        tilt = shoulder_tilt_deg(pose_lms, w, h)
        htilt = head_tilt_deg(pose_lms, w, h)
        face_w = None
        if face_lms is not None:
            face_w, _ = face_metrics(face_lms, w, h)

        # if calibrating
        if self.calibrating:
            if self.cal_start is None:
                self.cal_start = time.time()
            # require face visible 
            if pose_visibility_score(pose_lms) >= POSE_CONFIDENCE_LOW and face_w is not None:
                self.cal_samples.append((fh, tilt, htilt, face_w))
            elapsed = time.time() - self.cal_start
            if elapsed >= POSTURE_CALIBRATION_SECS and len(self.cal_samples) >= 5:
                fhs = [s[0] for s in self.cal_samples]
                tilts = [s[1] for s in self.cal_samples]
                htilts = [s[2] for s in self.cal_samples]
                fws = [s[3] for s in self.cal_samples]

                self.baseline_fh = float(np.median(fhs))
                self.baseline_tilt = float(np.median(tilts))
                self.baseline_head_tilt = float(np.median(htilts))
                self.baseline_face_width = float(np.median(fws))

                self.calibrating = False
                print(f"[POSTURE] calibration done: baseline_fh={self.baseline_fh:.3f}, "
                      f"baseline_tilt={self.baseline_tilt:.1f}deg, "
                      f"baseline_head_tilt={self.baseline_head_tilt:.1f}deg, "
                      f"baseline_face_width={self.baseline_face_width:.3f}")
            
            return "calibrating"   
        
        # calculating deviations from baseline
        fh_dev = abs(fh - self.baseline_fh)
        tilt_dev = abs(tilt - self.baseline_tilt)
        head_tilt_dev = abs(htilt - self.baseline_head_tilt)

        # "too close" = face currently much wider than baseline
        too_close = False
        if face_w is not None and self.baseline_face_width is not None:
            too_close = face_w >= self.baseline_face_width * FACE_CLOSE_RATIO

        # record reason for overlay / debugging
        reasons = []
        if fh_dev > FORWARD_HEAD_THRESHOLD:
            reasons.append("forward head")
        if tilt_dev > SHOULDER_TILT_THRESHOLD:
            reasons.append("shoulder tilt")
        if head_tilt_dev > HEAD_TILT_THRESHOLD:
            reasons.append("head tilt")
        if too_close:
            reasons.append("too close")

        is_bad = len(reasons) > 0
        self.last_reason = ", ".join(reasons) if reasons else None

        now = time.time()
        self.history.append((now, is_bad))
        cutoff = now - self.window
        self.history = [(t, b) for t, b in self.history if t >= cutoff]

        return "bad" if is_bad else "ok"    
    
    # what percent of frames are bad
    def score(self):
        if not self.history:
            return 0.0
        return sum(1 for _, b in self.history if b) / len(self.history)
    
    def seconds_of_data(self):
        if len(self.history) < 2:
            return 0.0
        return self.history[-1][0] - self.history[0][0]

    
#________________________________________________________
# TEXT + LANDMARKS OVERLAY HELPER FUNCTIONS
#_______________________________________________________
def put_text(frame, text, y, color=(255,255,255)):
    cv2.putText(frame,text,(10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)

def put_text_right(frame, text, y, color=(255,255,255)):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    x = frame.shape[1] - tw - 10
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

# upper body pipes between landmarks
POSE_CONNECTIONS_UPPER = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (NOSE, LEFT_EAR_POSE),
    (NOSE, RIGHT_EAR_POSE),
    (LEFT_EAR_POSE, LEFT_SHOULDER),
    (RIGHT_EAR_POSE, RIGHT_SHOULDER),]

# drawing face mesh dots and upper body pipes
def draw_landmarks(frame, face_lms, pose_lms, w, h):

    # face mesh
    if face_lms is not None:
        for lm in face_lms:
            x = int(lm.x * w)
            y = int(lm.y * h)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x,y), 1, (0, 255, 180), -1, cv2.LINE_AA)

    # upper body pose
    if pose_lms is not None:
        upper_idxs = [NOSE, LEFT_EAR_POSE, RIGHT_EAR_POSE, LEFT_SHOULDER, RIGHT_SHOULDER]

        # pipes between joints
        for a, b in POSE_CONNECTIONS_UPPER:
            la = pose_lms[a]
            lb = pose_lms[b]
            if la.visibility < 0.3 or lb.visibility < 0.3:
                continue
            pa = (int(la.x * w), int(la.y * h))
            pb = (int(lb.x * w), int(lb.y * h))
            cv2.line(frame, pa, pb, (255, 180, 0), 3, cv2.LINE_AA)

        # joint dots
        for i in upper_idxs:
            lm = pose_lms[i]
            if lm.visibility < 0.3:
                continue
            p = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, p, 6, (0, 0, 0), -1, cv2.LINE_AA)             # outline
            cv2.circle(frame, p, 5, (255, 255, 255), -1, cv2.LINE_AA)       # inside


# ________________________________________________________
# MAIN 
# ________________________________________________________
def main():
    
    # opening webcam
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)     # for windows specifically

    # trouble shooting
    if not cap.isOpened():
        print("[ERROR] camera couldn't be opened")
        sys.exit(1)

    # setting resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] press Q to quit or press S to screenshot")
    print(f"[INFO] PERCLOS window: {PERCLOS_WINDOW} s | threshold: {SLEEPY_THRESHOLD*100:.0f}%")
    print(f"[INFO] posture calibration: {POSTURE_CALIBRATION_SECS}s | bad threshold: {BAD_POSTURE_THRESHOLD * 100:.0f}%")

    # serial communication ------------------------------------------------
    bridge = SerialBridge(port=None)      # None = autodetect, can also type name manually (note: make sure arduino IDE serial monitor is not on before bridge connection goes through, otherwise the port is busy)
    bridge.start()

    perclos_tracker = PercloseTracker()
    posture_tracker = PostureTracker()
    posture_tracker.start_calibration()
    motor_signal = MotorSignal(bridge=bridge)

    sleepy = False
    prev_sleepy = False                 # for rising/falling edge detection on sleepy alerts
    alarm_start_time = None             # time.time() when alarm started sounding, or None if not sounding
    bad_posture_alert = False
    screenshot_count = 0
    fps = 0.0
    fps_time = time.time()
    eyes_closed = False                 # persists across frames so hysteresis works

    while True:

        # capturing frames -----------------------------
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] frame grab failed")
            break

        h, w = frame.shape[:2]

        # downscale frames for the models
        small = cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # FPS
        now_fps = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (now_fps - fps_time + 1e-6))
        fps_time = now_fps

        # beep boop beep boop running model O_0 -------------------------------
        face_results = face_landmarker.detect(mp_image)
        pose_results = pose_landmarker.detect(mp_image)

        pose_lms = pose_results.pose_landmarks[0] if pose_results.pose_landmarks else None
        face_lms = face_results.face_landmarks[0] if face_results.face_landmarks else None

        # PERCLOS calculations ------------------- 
        ear_avg = None

        # EAR calculation
        if face_lms:
            left_ear = eye_aspect_ratio(face_lms, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(face_lms, RIGHT_EYE, w, h)
            ear_avg = (left_ear + right_ear)/2.0
            if eyes_closed:
                eyes_closed = ear_avg < EAR_OPEN_THRESHOLD      # stay closed until clearly eyes were opened
            else:
                eyes_closed = ear_avg < EAR_THRESHOLD           # normal threshold
            perclos_tracker.update(eyes_closed)

        perclos = perclos_tracker.score()
        perclos_secs = perclos_tracker.seconds_of_data()

        # sleepy detection + alarm logic ---------------------------------

        # wake detection: 
        if sleepy and alarm_start_time is not None:
            alarm_duration = time.time() - alarm_start_time
            if alarm_duration >= WAKE_MIN_ALARM_SECS and \
               perclos_tracker.recent_open_ratio(WAKE_WINDOW_SECS) >= WAKE_OPEN_RATIO:      # wait a bit, (time for user to react to the alarm)
                perclos_tracker.reset()
                perclos = perclos_tracker.score()
                perclos_secs = perclos_tracker.seconds_of_data()
                print(f"[INFO] wake detected after {alarm_duration:.1f}s, resetting PERCLOS history")

        prev_sleepy = sleepy    # snapshot BEFORE we reassign sleepy below

        if perclos_secs >= 10 and perclos >= SLEEPY_THRESHOLD:
            if not sleepy:
                alarm_start_time = time.time()     # record when the alarm first fired
            sleepy = True
            start_alarm()
        else:
            sleepy = False
            alarm_start_time = None                # reset
            stop_alarm()

        # posture checking ---------------------------
        posture_state = posture_tracker.update(pose_lms, face_lms, w, h)
        posture_score = posture_tracker.score()
        posture_secs = posture_tracker.seconds_of_data()

        prev_bad_posture_alert = bad_posture_alert
        if posture_secs >= 5.0 and posture_score >= BAD_POSTURE_THRESHOLD:
            bad_posture_alert = True
        else:
            bad_posture_alert = False

        # rising edge -> fire the one-shot chime 
        if bad_posture_alert and not prev_bad_posture_alert:
            trigger_posture_chime()
            reason = posture_tracker.last_reason or "posture"
            print(f"[POSTURE] bad posture alert ({reason})")

        # motor signals ---------------------------------------------------
        motor_signal.update(pose_lms, face_lms, w, h)

        # teensy communication --------------------------------------------
        if sleepy != prev_sleepy:   # edge detection
            bridge.send_alert("sleepy", on=sleepy)
        if bad_posture_alert != prev_bad_posture_alert:
            bridge.send_alert("posture", on=bad_posture_alert,
                              reason=posture_tracker.last_reason)

        # status coalesced inside the bridge
        if posture_state == "calibrating":
            state_str = "calibrating"
        elif sleepy:
            state_str = "sleepy"
        elif bad_posture_alert:
            state_str = "bad_posture"
        else:
            state_str = "ok"

        bridge.send_status(perclos=perclos, posture=posture_score,
                           state=state_str, ear=ear_avg)

        # overlay ------------------------------------------------------

        # landmarks
        draw_landmarks(frame, face_lms, pose_lms, w, h)

        # perclos
        if ear_avg is not None:
            put_text(frame, f"EAR: {ear_avg:.3f}", 30)
            put_text(frame, f"Eyes: {'CLOSED' if eyes_closed else 'open'}", 60, (0, 80, 255) if eyes_closed else (255, 255, 255))
        else:
            put_text(frame, "EAR: no face detected", 30, (100, 100, 100))
            put_text(frame, "Eyes: ------", 60, (100, 100, 100))
        put_text(frame, f"PERCLOS: {perclos*100:.1f}%", 90, (0, 80, 255) if sleepy else (255, 255, 255))

        if perclos_secs < 10:
            put_text(frame, "calibrating...", 120, (180, 180, 0))
        else:
            put_text(frame, f"Status: {'SLEEPY!' if sleepy else 'ok'}", 120, (0, 80, 255) if sleepy else (255, 255, 255))

        # FPS
        put_text(frame, f"FPS: {fps:.1f}", 150, (180, 180, 0) if fps < 15 else (255, 255, 255))

        # face metrics (for tuning TOO_CLOSE_FACE_WIDTH_RATIO / FACE_Y thresholds)
        if face_lms is not None:
            fw_dbg, fcy_dbg = face_metrics(face_lms, w, h)
            put_text(frame, f"face w: {fw_dbg:.2f}  y: {fcy_dbg:.2f}", 180, (180, 180, 180))

        # posture
        if posture_state == 'calibrating':
            elapsed_cal = time.time() - posture_tracker.cal_start if posture_tracker.cal_start else 0
            remaining = max(0, POSTURE_CALIBRATION_SECS - elapsed_cal)
            put_text_right(frame, f"POSTURE: calibrating ({remaining:.0f}s)", 30, (180, 180, 0))
        elif posture_state == "no_pose":
            put_text_right(frame, "POSTURE: no body detected", 30, (100, 100, 100))
        else:
            frame_color = (0, 80, 255) if posture_state == "bad" else (255, 255, 255)
            put_text_right(frame, f"current frame: {'BAD' if posture_state == 'bad' else 'ok'}", 30, frame_color)           # current score
            put_text_right(frame, f"POSTURE: {posture_score*100:.0f}% bad ({posture_secs:.0f}s)", 60, (255, 255, 255))      # rolling percentage
            alert_color = (0, 80, 255) if bad_posture_alert else (255, 255, 255)
            put_text_right(frame, f"POSTURE: {'BAD POSTURE!' if bad_posture_alert else 'ok'}", 90, alert_color)             # above/below threshold
            if posture_state == "bad" and posture_tracker.last_reason:
                put_text_right(frame, f"reason: {posture_tracker.last_reason}", 120, (0, 80, 255))

        # motor signals ------------------------------------------------
        if motor_signal.active_signals:
            y_sig = h - 20 - (len(motor_signal.active_signals) - 1) * 28
            for sig in motor_signal.active_signals:
                put_text_right(frame, sig, y_sig, (157, 46, 230))
                y_sig += 28

        cv2.imshow(WINDOW_NAME, frame)

        # handling key presses ---------------------------------------------------
        key = cv2.waitKey(1) & 0xFF     # default -1

        if key == ord('q') or key == ord('Q'):
            print("[INFO] quitting")
            break

        elif key == ord('s') or key == ord('S'):
            screenshot_count += 1
            filename = f"screenshot_{screenshot_count}.png"
            cv2.imwrite(filename, frame)
            print(f"[INFO] screenshot saved as {filename}")

        # manual recalibration
        elif key == ord('r') or key == ord('R'):
            posture_tracker.start_calibration()
            bridge.send_calibrate()        # let the teensy know so the LCD updates
            print("[INFO] posture recalibration triggered")

    # closing
    stop_alarm()
    bridge.stop()
    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    pose_landmarker.close()

if __name__ == "__main__":
    main()
