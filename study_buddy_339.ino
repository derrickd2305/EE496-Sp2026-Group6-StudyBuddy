// USC EE496 | Group 6 | Study Buddy 
// Teensy 4.1 firmware
// flowa powa

#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <SerialFlash.h>
#include <ILI9341_t3.h>
#include <Servo.h>
#include <Adafruit_LSM6DS3TRC.h>


// ============================================================================
// PINOUT
// ============================================================================

constexpr uint8_t TFT_CS         = 10;
constexpr uint8_t TFT_DC         = 9;
constexpr uint8_t TFT_RST        = 8;
constexpr uint8_t SERVO_PIN      = 5;
constexpr uint8_t ACT_IN1_PIN    = 3;
constexpr uint8_t ACT_IN2_PIN    = 4;

// ============================================================================
// LCD
// ============================================================================

ILI9341_t3 tft(TFT_CS, TFT_DC, TFT_RST);
constexpr int LCD_W  = 320;
constexpr int LCD_H  = 240;
constexpr int FACE_CX = 160;
constexpr int FACE_CY = 110;
constexpr int FACE_R  = 60;

// ============================================================================
// AUDIO
// ============================================================================

AudioSynthWaveformSine sine1;
AudioPlaySdWav         playWav;
AudioMixer4            mixerL;
AudioMixer4            mixerR;
AudioOutputI2S         i2sOut;

AudioConnection patchSineL(sine1,    0, mixerL, 0);
AudioConnection patchSineR(sine1,    0, mixerR, 0);
AudioConnection patchWavL (playWav,  0, mixerL, 1);
AudioConnection patchWavR (playWav,  1, mixerR, 1);
AudioConnection patchOutL (mixerL,   0, i2sOut, 0);
AudioConnection patchOutR (mixerR,   0, i2sOut, 1);

float    master_volume_ = 0.5f;
constexpr float TONE_GAIN_REL = 1.0f;
constexpr float WAV_GAIN_REL  = 1.0f;

bool sd_ok_       = false;
bool wav_playing_ = false;

enum AudioPattern { PAT_NONE, PAT_ALARM, PAT_CHIME };
AudioPattern audio_pattern_   = PAT_NONE;
uint8_t      pattern_step_    = 0;
uint32_t     pattern_next_ms_ = 0;

// ============================================================================
// SERVO (yaw)
// ============================================================================

Servo yaw_servo;
constexpr int YAW_PULSE_MIN_US = 750;
constexpr int YAW_PULSE_MAX_US = 2250;
constexpr int YAW_MIN_DEG      = 0;
constexpr int YAW_MAX_DEG      = 180;
int yaw_angle_ = 90;

// ============================================================================
// ACTUATOR (height)
// ============================================================================

constexpr uint32_t ACT_FULL_STROKE_MS    = 3500;       // time in ms for full extension
constexpr uint32_t ACT_HOMING_DURATION   = 5000;       
int32_t  act_position_ms_      = 0;
uint32_t act_dir_started_      = 0;
int8_t   act_direction_        = 0;                    // -1 retract, 0 stop, +1 extend
uint32_t act_stop_at_ms_       = 0;                    // 0 = no scheduled stop
bool     act_homed_            = false;

// ============================================================================
// IMU
// ============================================================================

Adafruit_LSM6DS3TRC imu;
bool imu_ok_ = false;

// IMU sampling
constexpr uint32_t IMU_SAMPLE_PERIOD_MS = 20;          // 50 Hz
uint32_t imu_last_sample_ms_ = 0;
float    imu_pitch_deg_ = 0.0f;
float    imu_roll_deg_  = 0.0f;
float    imu_gyro_z_    = 0.0f;

// IMU streaming to host
constexpr uint32_t IMU_STREAM_PERIOD_MS = 50;          // 20 Hz
uint32_t imu_last_stream_ms_ = 0;
bool     imu_streaming_ = false;                       // toggle via "IMU_STREAM:ON|OFF"

// Closed-loop yaw verification - off by default until IMU is on the head
constexpr bool     YAW_VERIFY_ENABLED   = false;
constexpr uint32_t YAW_VERIFY_WINDOW_MS = 500;
constexpr float    YAW_GYRO_THRESHOLD   = 0.3f;        // rad/s peak during motion

// ============================================================================
// LCD STATE
// ============================================================================

struct LcdState {
  String  status      = "boot";
  int     perclos_pct = -1;
  int     posture_pct = -1;
  String  msg         = "";
  String  drawn_status      = "<<unset>>";
  int     drawn_perclos_pct = -2;
  int     drawn_posture_pct = -2;
  String  drawn_msg         = "<<unset>>";
} lcd;

// ============================================================================
constexpr uint32_t STATE_INTERVAL_MS = 200;            // 5 Hz
uint32_t last_state_ms_ = 0;

void apply_volume();
void start_alarm();
void stop_alarm();
void trigger_chime();
void audio_pattern_tick();
bool start_wav(const String& fname);
void stop_wav();
void servo_set_angle(int deg);
void servo_set_relative(int delta);
void actuator_drive(int8_t dir, uint32_t ms);
void actuator_stop();
void actuator_home_blocking();
void actuator_tick();
void actuator_accrue_travel();
void imu_tick();
void lcd_full_redraw();
void lcd_tick();
void draw_face(const String& state);
void draw_metrics_row();
void draw_msg_row();
void parse_line(const String& line);
void send_state();
void send_imu();

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && (millis() - t0) < 1500) {}

  // -- audio ------------
  AudioMemory(12);
  sine1.amplitude(0.0f);
  sine1.frequency(440);
  apply_volume();

  // --- LCD ------
  tft.begin();
  tft.setRotation(1);
  lcd_full_redraw();

  // --- SD ----------------------------------
  if (SD.begin(BUILTIN_SDCARD)) {
    sd_ok_ = true;
    Serial.println("LOG:sd ok");
  } else {
    sd_ok_ = false;
    Serial.println("LOG:sd init failed (wav playback disabled)");
  }

  // --- IMU ------------------------------------------------
  if (imu.begin_I2C()) {
    imu.setAccelRange(LSM6DS_ACCEL_RANGE_4_G);
    imu.setGyroRange(LSM6DS_GYRO_RANGE_500_DPS);
    imu.setAccelDataRate(LSM6DS_RATE_104_HZ);
    imu.setGyroDataRate(LSM6DS_RATE_104_HZ);
    imu_ok_ = true;
    Serial.println("LOG:imu ok");
  } else {
    imu_ok_ = false;
    Serial.println("LOG:imu init failed");
  }

  // --- servo -----------------------------------------------------
  yaw_servo.attach(SERVO_PIN, YAW_PULSE_MIN_US, YAW_PULSE_MAX_US);
  servo_set_angle(90);
  delay(500);        // settle

  // --- actuator ----------------------------------------------------
  pinMode(ACT_IN1_PIN, OUTPUT);
  pinMode(ACT_IN2_PIN, OUTPUT);
  digitalWrite(ACT_IN1_PIN, LOW);
  digitalWrite(ACT_IN2_PIN, LOW);
  actuator_home_blocking();

  Serial.println("BOOT");
}

// ============================================================================
// LOOP
// ============================================================================

String rx_buf_;

void loop() {
  // 1. drain serial input
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      parse_line(rx_buf_);
      rx_buf_ = "";
    } else if (rx_buf_.length() < 120) {
      rx_buf_ += c;
    }
  }

  // 2. audio state machine
  audio_pattern_tick();

  // 3. wav playback completion
  if (wav_playing_ && !playWav.isPlaying()) {
    wav_playing_ = false;
    Serial.println("LOG:wav done");
  }

  // 4. actuator timed-stop
  actuator_tick();

  // 5. IMU sampling
  imu_tick();

  // 6. LCD zone updates
  lcd_tick();

  // 7. periodic STATE message
  uint32_t now = millis();
  if (now - last_state_ms_ >= STATE_INTERVAL_MS) {
    last_state_ms_ = now;
    send_state();
  }
}

// ============================================================================
// AUDIO
// ============================================================================

// master volume applies to all volumes (don't think this is fully necessary, but robust for stereo)
void apply_volume() {
  float v = master_volume_;
  if (v < 0.0f) v = 0.0f;
  if (v > 1.0f) v = 1.0f;
  mixerL.gain(0, v * TONE_GAIN_REL);
  mixerR.gain(0, v * TONE_GAIN_REL);
  mixerL.gain(1, v * WAV_GAIN_REL);
  mixerR.gain(1, v * WAV_GAIN_REL);
}

void start_alarm() {
  audio_pattern_   = PAT_ALARM;
  pattern_step_    = 0;
  pattern_next_ms_ = millis();
}

void stop_alarm() {
  audio_pattern_ = PAT_NONE;
  sine1.amplitude(0.0f);
}

void trigger_chime() {
  if (audio_pattern_ == PAT_ALARM) {
    Serial.println("LOG:chime skipped (alarm active)");
    return;
  }
  audio_pattern_   = PAT_CHIME;
  pattern_step_    = 0;
  pattern_next_ms_ = millis();
}

// simple alarms (used for debugging, disabled for final, using wav files instead)
void audio_pattern_tick() {
  uint32_t now = millis();
  if (audio_pattern_ == PAT_NONE) return;
  if ((int32_t)(now - pattern_next_ms_) < 0) return;

  if (audio_pattern_ == PAT_ALARM) {
    switch (pattern_step_ % 6) {
      case 0: case 2: case 4:
        sine1.frequency(880);
        sine1.amplitude(0.6f);
        pattern_next_ms_ = now + 150;
        break;
      case 1: case 3:
        sine1.amplitude(0.0f);
        pattern_next_ms_ = now + 150;
        break;
      case 5:
        sine1.amplitude(0.0f);
        pattern_next_ms_ = now + 700;
        break;
    }
    pattern_step_++;
    return;
  }

  if (audio_pattern_ == PAT_CHIME) {
    switch (pattern_step_) {
      case 0: sine1.frequency(600); sine1.amplitude(0.5f); pattern_next_ms_ = now + 250; break;
      case 1: sine1.amplitude(0.0f);                       pattern_next_ms_ = now + 50;  break;
      case 2: sine1.frequency(450); sine1.amplitude(0.5f); pattern_next_ms_ = now + 250; break;
      case 3: sine1.amplitude(0.0f); audio_pattern_ = PAT_NONE;                          break;
    }
    pattern_step_++;
    return;
  }
}

bool start_wav(const String& fname) {
  if (!sd_ok_) {
    Serial.println("ERR:no_sd");
    return false;
  }
  if (playWav.isPlaying()) playWav.stop();
  if (!playWav.play(fname.c_str())) {
    Serial.print("ERR:wav_play_failed:");
    Serial.println(fname);
    return false;
  }
  delay(5);
  wav_playing_ = true;
  return true;
}

void stop_wav() {
  if (playWav.isPlaying()) playWav.stop();
  wav_playing_ = false;
}

// ============================================================================
// SERVO
// ============================================================================

void servo_set_angle(int deg) {
  if (deg < YAW_MIN_DEG) deg = YAW_MIN_DEG;
  if (deg > YAW_MAX_DEG) deg = YAW_MAX_DEG;
  yaw_servo.write(YAW_MAX_DEG - deg);
  yaw_angle_ = deg;

  // optional closed-loop verification via IMU gyro Z (off right now)
  if (YAW_VERIFY_ENABLED && imu_ok_) {
    uint32_t t_end = millis() + YAW_VERIFY_WINDOW_MS;
    float    peak  = 0.0f;
    while (millis() < t_end) {
      sensors_event_t a, g, t;
      imu.getEvent(&a, &g, &t);
      float gz = fabsf(g.gyro.z);
      if (gz > peak) peak = gz;
      delay(5);
    }
    if (peak < YAW_GYRO_THRESHOLD) {
      Serial.print("LOG:yaw_verify_fail peak_gz=");
      Serial.println(peak, 3);
    }
  }
}

void servo_set_relative(int delta) {
  servo_set_angle(yaw_angle_ + delta);
}

// ============================================================================
// ACTUATOR
// ============================================================================

void actuator_set_outputs(int8_t dir) {
  switch (dir) {
    case +1:  digitalWrite(ACT_IN1_PIN, HIGH); digitalWrite(ACT_IN2_PIN, LOW);  break;
    case -1:  digitalWrite(ACT_IN1_PIN, LOW);  digitalWrite(ACT_IN2_PIN, HIGH); break;
    default:  digitalWrite(ACT_IN1_PIN, LOW);  digitalWrite(ACT_IN2_PIN, LOW);  break;
  }
}

void actuator_accrue_travel() {
  if (act_direction_ == 0) return;
  uint32_t elapsed = millis() - act_dir_started_;
  act_position_ms_ += (int32_t)act_direction_ * (int32_t)elapsed;
  if (act_position_ms_ < 0)                          act_position_ms_ = 0;
  if (act_position_ms_ > (int32_t)ACT_FULL_STROKE_MS) act_position_ms_ = ACT_FULL_STROKE_MS;
  act_dir_started_ = millis();
}

void actuator_stop() {
  actuator_accrue_travel();
  actuator_set_outputs(0);
  act_direction_  = 0;
  act_stop_at_ms_ = 0;
}

// direction, time in ms
void actuator_drive(int8_t dir, uint32_t ms) {
  if (dir != +1 && dir != -1) {
    actuator_stop();
    return;
  }

  // soft limit refusal (limit switches)
  actuator_accrue_travel();
  if (dir == +1 && act_position_ms_ >= (int32_t)ACT_FULL_STROKE_MS) {
    Serial.println("LOG:actuator already at extend limit");
    return;
  }
  if (dir == -1 && act_position_ms_ <= 0) {
    Serial.println("LOG:actuator already at retract limit");
    return;
  }

  actuator_set_outputs(dir);
  act_direction_   = dir;
  act_dir_started_ = millis();
  act_stop_at_ms_  = act_dir_started_ + ms;
}

void actuator_tick() {
  if (act_direction_ == 0) return;
  uint32_t now = millis();

  // scheduled stop reached?
  if (act_stop_at_ms_ != 0 && (int32_t)(now - act_stop_at_ms_) >= 0) {
    actuator_stop();
    return;
  }

  // check if we reached soft limit
  uint32_t elapsed   = now - act_dir_started_;
  int32_t  projected = act_position_ms_ + (int32_t)act_direction_ * (int32_t)elapsed;
  bool overshoot = false;
  if (act_direction_ > 0 && projected > (int32_t)ACT_FULL_STROKE_MS) overshoot = true;
  if (act_direction_ < 0 && projected < 0)                            overshoot = true;
  if (overshoot) {
    Serial.println("LOG:actuator soft limit");
    actuator_stop();
  }
}

void actuator_home_blocking() {
  Serial.println("LOG:homing actuator");
  actuator_set_outputs(-1);
  act_direction_   = -1;
  act_dir_started_ = millis();
  act_stop_at_ms_  = 0;
  delay(ACT_HOMING_DURATION);
  actuator_set_outputs(0);
  act_direction_   = 0;
  act_position_ms_ = 0;
  act_homed_       = true;
  Serial.println("LOG:actuator homed");
}

// ============================================================================
// IMU
// ============================================================================

void imu_tick() {
  if (!imu_ok_) return;
  uint32_t now = millis();
  if (now - imu_last_sample_ms_ < IMU_SAMPLE_PERIOD_MS) return;
  imu_last_sample_ms_ = now;

  sensors_event_t a, g, t;
  imu.getEvent(&a, &g, &t);

  float ax = a.acceleration.x;
  float ay = a.acceleration.y;
  float az = a.acceleration.z;
  imu_pitch_deg_ = atan2f(-ax, sqrtf(ay*ay + az*az)) * 180.0f / PI;
  imu_roll_deg_  = atan2f( ay, az)                   * 180.0f / PI;
  imu_gyro_z_    = g.gyro.z;

  if (imu_streaming_ && (now - imu_last_stream_ms_) >= IMU_STREAM_PERIOD_MS) {
    imu_last_stream_ms_ = now;
    send_imu();
  }
}

void send_imu() {
  Serial.print("IMU:P=");
  Serial.print(imu_pitch_deg_, 1);
  Serial.print(",R=");
  Serial.print(imu_roll_deg_, 1);
  Serial.print(",GZ=");
  Serial.println(imu_gyro_z_, 3);
}

// ============================================================================
// LCD
// ============================================================================

void lcd_full_redraw() {
  tft.fillScreen(ILI9341_BLACK);
  lcd.drawn_status      = "<<force>>";
  lcd.drawn_perclos_pct = -2;
  lcd.drawn_posture_pct = -2;
  lcd.drawn_msg         = "<<force>>";
  lcd_tick();
}

void lcd_tick() {
  if (lcd.status != lcd.drawn_status) {
    draw_face(lcd.status);
    lcd.drawn_status = lcd.status;
  }
  if (lcd.perclos_pct != lcd.drawn_perclos_pct ||
      lcd.posture_pct != lcd.drawn_posture_pct) {
    draw_metrics_row();
    lcd.drawn_perclos_pct = lcd.perclos_pct;
    lcd.drawn_posture_pct = lcd.posture_pct;
  }
  if (lcd.msg != lcd.drawn_msg) {
    draw_msg_row();
    lcd.drawn_msg = lcd.msg;
  }
}

uint16_t border_color_for(const String& s) {
  if (s == "sleepy")       return ILI9341_RED;
  if (s == "bad_posture")  return ILI9341_YELLOW;
  if (s == "calibrating")  return ILI9341_CYAN;
  if (s == "nopose")       return 0x7BEF;
  return ILI9341_GREEN;
}

void draw_face(const String& state) {
  tft.fillRect(0, 0, LCD_W, 190, ILI9341_BLACK);

  uint16_t border = border_color_for(state);
  tft.drawRect(0, 0, LCD_W, LCD_H, border);
  tft.drawRect(1, 1, LCD_W - 2, LCD_H - 2, border);

  tft.fillCircle(FACE_CX, FACE_CY, FACE_R, 0xFFE0);

  if (state == "sleepy") {
    for (int i = -2; i <= 2; i++) {
      tft.drawLine(FACE_CX - 32, FACE_CY - 15 + i, FACE_CX - 12, FACE_CY - 15 + i, ILI9341_BLACK);
      tft.drawLine(FACE_CX + 12, FACE_CY - 15 + i, FACE_CX + 32, FACE_CY - 15 + i, ILI9341_BLACK);
    }
    tft.drawCircle(FACE_CX, FACE_CY + 25, 7, ILI9341_BLACK);
    tft.setTextColor(ILI9341_WHITE);
    tft.setTextSize(2);
    tft.setCursor(235, 60); tft.print('z');
    tft.setCursor(250, 45); tft.print('z');
    tft.setCursor(265, 30); tft.print('z');
  } else if (state == "bad_posture") {
    tft.fillCircle(FACE_CX - 23, FACE_CY - 17, 5, ILI9341_BLACK);
    tft.fillCircle(FACE_CX + 23, FACE_CY - 17, 5, ILI9341_BLACK);
    tft.drawLine(FACE_CX - 35, FACE_CY - 32, FACE_CX - 12, FACE_CY - 28, ILI9341_BLACK);
    tft.drawLine(FACE_CX + 12, FACE_CY - 28, FACE_CX + 35, FACE_CY - 32, ILI9341_BLACK);
    for (int i = 0; i < 3; i++) {
      tft.drawLine(FACE_CX - 20, FACE_CY + 22 - i, FACE_CX + 20, FACE_CY + 22 - i, ILI9341_BLACK);
    }
  } else if (state == "calibrating") {
    tft.fillCircle(FACE_CX - 23, FACE_CY - 17, 5, ILI9341_BLACK);
    tft.fillCircle(FACE_CX + 23, FACE_CY - 17, 5, ILI9341_BLACK);
    tft.drawLine(FACE_CX - 18, FACE_CY + 18, FACE_CX + 18, FACE_CY + 18, ILI9341_BLACK);
    tft.setTextColor(ILI9341_CYAN);
    tft.setTextSize(2);
    tft.setCursor(FACE_CX - 60, 175);
    tft.print("calibrating");
  } else if (state == "nopose") {
    tft.fillCircle(FACE_CX - 23, FACE_CY - 17, 5, ILI9341_BLACK);
    tft.fillCircle(FACE_CX + 23, FACE_CY - 17, 5, ILI9341_BLACK);
    tft.drawLine(FACE_CX - 10, FACE_CY + 18, FACE_CX + 10, FACE_CY + 18, ILI9341_BLACK);
    tft.setTextColor(0x7BEF);
    tft.setTextSize(2);
    tft.setCursor(FACE_CX - 60, 175);
    tft.print("no pose");
  } else {
    // ok / happy
    tft.fillCircle(FACE_CX - 23, FACE_CY - 17, 7, ILI9341_BLACK);
    tft.fillCircle(FACE_CX + 23, FACE_CY - 17, 7, ILI9341_BLACK);
    for (int i = 0; i < 3; i++) {
      tft.drawLine(FACE_CX - 20, FACE_CY + 15 + i, FACE_CX,      FACE_CY + 28 + i, ILI9341_BLACK);
      tft.drawLine(FACE_CX,      FACE_CY + 28 + i, FACE_CX + 20, FACE_CY + 15 + i, ILI9341_BLACK);
    }
  }
}

void draw_metrics_row() {
  tft.fillRect(8, 195, LCD_W - 16, 18, ILI9341_BLACK);
  tft.setTextSize(1);
  tft.setTextColor(ILI9341_WHITE);
  tft.setCursor(10, 200);
  tft.print("PERCLOS:");
  if (lcd.perclos_pct >= 0) { tft.print(lcd.perclos_pct); tft.print('%'); }
  else                       { tft.print("--"); }
  tft.setCursor(160, 200);
  tft.print("POSTURE:");
  if (lcd.posture_pct >= 0) { tft.print(lcd.posture_pct); tft.print('%'); }
  else                       { tft.print("--"); }
}

void draw_msg_row() {
  tft.fillRect(8, 218, LCD_W - 16, 16, ILI9341_BLACK);
  tft.setTextSize(1);
  tft.setTextColor(ILI9341_WHITE);
  tft.setCursor(10, 222);
  String m = lcd.msg.length() > 50 ? lcd.msg.substring(0, 50) : lcd.msg;
  tft.print(m);
}

// ============================================================================
// PROTOCOL PARSING
// ============================================================================

static void ack(const String& verb)   { Serial.print("ACK:"); Serial.println(verb); }
static void err(const String& reason) { Serial.print("ERR:"); Serial.println(reason); }

void parse_line(const String& line) {
  if (line.length() == 0) return;

  int colon = line.indexOf(':');
  String verb = (colon < 0) ? line : line.substring(0, colon);
  String rest = (colon < 0) ? ""   : line.substring(colon + 1);
  verb.trim();
  rest.trim();

  // ------ bare verbs -----------------------------
  if (verb == "PING") { Serial.println("PONG"); return; }

  if (verb == "RESET") {
    stop_alarm(); stop_wav();
    actuator_stop();
    servo_set_angle(90);
    lcd.status = "ok"; lcd.perclos_pct = -1; lcd.posture_pct = -1; lcd.msg = "";
    ack("RESET");
    return;
  }

  if (verb == "CHIME")        { trigger_chime();   ack("CHIME");    return; }
  if (verb == "WAV_STOP")     { stop_wav();        ack("WAV_STOP"); return; }
  if (verb == "HEIGHT_STOP")  { actuator_stop();   ack("HEIGHT_STOP"); return; }
  if (verb == "HEIGHT_HOME")  { actuator_home_blocking(); ack("HEIGHT_HOME"); return; }

  // ---- ALARM:ON/OFF --------------------------------------------
  if (verb == "ALARM") {
    if (rest == "ON")  { start_alarm(); ack("ALARM"); return; }
    if (rest == "OFF") { stop_alarm();  ack("ALARM"); return; }
    err("bad_args"); return;
  }

  // --------------------------- VOL:[0-100] --------------------------------
  if (verb == "VOL") {
    int v = rest.toInt();
    if (v < 0 || v > 100) { err("bad_args"); return; }
    master_volume_ = v / 100.0f;
    apply_volume();
    ack("VOL");
    return;
  }

  // WAV:<filename> ----------------------------------------------------------
  if (verb == "WAV") {
    if (rest.length() == 0) { err("bad_args"); return; }
    if (start_wav(rest)) ack("WAV");
    return;
  }

  // YAW:<deg> ---------------------------------------------------------------
  if (verb == "YAW") {
    int a = rest.toInt();
    if (a < YAW_MIN_DEG || a > YAW_MAX_DEG) { err("bad_args"); return; }
    servo_set_angle(a);
    ack("YAW");
    return;
  }

  //YAW_REL:<delta> ------------------------------------------------------------
  if (verb == "YAW_REL") {
    int d = rest.toInt();
    // sanity check: don't allow garbage from a parse failure on empty rest
    if (rest.length() == 0) { err("bad_args"); return; }
    servo_set_relative(d);
    ack("YAW_REL");
    return;
  }

  // HEIGHT_MS:<+-N> -----------------------------------------------------------
  if (verb == "HEIGHT_MS") {
    int ms = rest.toInt();
    if (rest.length() == 0 || ms == 0) { err("bad_args"); return; }
    int8_t dir = (ms > 0) ? +1 : -1;
    uint32_t magnitude = (uint32_t)(ms > 0 ? ms : -ms);
    if (magnitude > 10000) { err("bad_args"); return; }     // cap at 10s
    actuator_drive(dir, magnitude);
    ack("HEIGHT_MS");
    return;
  }

  // ---- IMU_STREAM:ON/OFF  -------------------
  if (verb == "IMU_STREAM") {
    if (rest == "ON")  { imu_streaming_ = true;  ack("IMU_STREAM"); return; }
    if (rest == "OFF") { imu_streaming_ = false; ack("IMU_STREAM"); return; }
    err("bad_args"); return;
  }

  // ------ LCD:* ---------
  if (verb == "LCD") {
    int colon2 = rest.indexOf(':');
    if (colon2 < 0) { err("bad_args"); return; }
    String sub = rest.substring(0, colon2);
    String val = rest.substring(colon2 + 1);
    sub.trim(); val.trim();

    if (sub == "STATUS") {
      if (val == "ok" || val == "sleepy" || val == "bad_posture" ||
          val == "calibrating" || val == "nopose") {
        lcd.status = val;
        ack("LCD_STATUS");
      } else { err("bad_args"); }
      return;
    }
    if (sub == "PERCLOS") {
      int p = val.toInt();
      if (p < 0 || p > 100) { err("bad_args"); return; }
      lcd.perclos_pct = p; ack("LCD_PERCLOS"); return;
    }
    if (sub == "POSTURE") {
      int p = val.toInt();
      if (p < 0 || p > 100) { err("bad_args"); return; }
      lcd.posture_pct = p; ack("LCD_POSTURE"); return;
    }
    if (sub == "MSG") {
      lcd.msg = val; ack("LCD_MSG"); return;
    }
    err("bad_args"); return;
  }

  err("unknown_cmd");
}

// ======================================================================================
// STATE
// ======================================================================================

void send_state() {
  // STATE:ALARM=<0|1>,VOL=<0-100>,WAV=<0|1>,YAW=<deg>,H=<ms>,IMU=<0|1>
  Serial.print("STATE:ALARM=");
  Serial.print(audio_pattern_ == PAT_ALARM ? 1 : 0);
  Serial.print(",VOL=");
  Serial.print((int)(master_volume_ * 100));
  Serial.print(",WAV=");
  Serial.print(wav_playing_ ? 1 : 0);
  Serial.print(",YAW=");
  Serial.print(yaw_angle_);
  Serial.print(",H=");
  // accrue travel so the H field is current even mid-motion
  actuator_accrue_travel();
  Serial.print(act_position_ms_);
  Serial.print(",IMU=");
  Serial.println(imu_ok_ ? 1 : 0);
}
