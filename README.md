# Study Buddy
EE496 Capstone Project | Spring 2026 | Group 6: Derrick Kim, Jojo Ibalio, Saniya Brown

A real-time drowsiness and posture monitoring system designed to help students develop healthier study habits. 
Study Buddy uses computer vision to track PERCLOS and upper-body posture, alerting the user through audio and visual cues, all in a user friendly package.
---
# How it works:
1. A python script on the host's laptop captures webcam footage and runs two MediaPipe models on each frame (face mesh + pose landmarker)
2. Metrics are derived from landmarks (PERCLOS, posture deviation, alert state)
3. Messages are sent through the serial bridge protocol to the Teensy 4.1
4. The Teensy microcontroller drives a TFT LCD display, a speaker, a servo, and a linear actuator
---
# Repository Structure
- study_buddy.py			# computer vision, landmark processing
- serial_comm.py			# serial communication protocol, bridge between host and Teensy
- study_buddy.ino   	# Teensy v4.1 firmware  
- PERCLOSCHIME.WAV   	# chime saved onto Teensy microSD card
- POSTURECHIME.WAV   	# chime saved onto Teensy microSD card
---
# Requirements
The current code was tested on Python 3.10, Windows (some lines in study_buddy.py must be modified for non-Windows OS)

## Python Environment
```
pip install opencv-python mediapipe==0.10.30 pyserial numpy
```

## Arduino IDE
1. Install Teensyduino on top of Arduino IDE
2. Download the ILI9341_t3 and Adafruit LSM6DS packages (all other necessary packages are dependencies of these two libraries)
---
# Hardware
- Teensy 4.1
- ILI9341 TFT LCD 3.2"
- MAX98357 I2S amplifier
- microSD card
- SG-5010 servo
- L298N linear actuator
- LSM6DS3TR-C IMU
---
# User Operation
1. Connect the 12V power supply to the barrel jack. Connect a USB-C cable between the Study Buddy’s USB-C port and the laptop. Check for the H-Bridge LED indicator to light red.
2. The user should ensure that the Teensy’s microSD card is loaded with the correct WAV files. The current names are PERCLOSCHIME.WAV and POSTURECHIME.WAV. If the user wants to change any of the files or file names, ensure that the wave files are encoded in signed 16-bit PCM, sampled at 44,100 Hz, and that the file names are in all capital letters. All of these can easily be changed through Audacity for free.
3. Ensure that the following libraries are installed for the Arduino IDE and the Python environment.
   - Arduino IDE: Install Teensyduino, then download ILI9341_t3 and Adafruit LSM6DS (all other necessary 					libraries are dependencies for these two libraries).
   - Python: pip install opencv-python mediapipe==0.10.30 pyserial numpy (remaining imports are all standard 			Python libraries).
4. Connect to the Teensy 4.1 USB port and flash the firmware from the Arduino IDE. Press the program button on the Teensy if prompted.
5. Verify hardware:
   - If not already, the startup sequence should have moved the servo to the center (YAW: 90), and the user can type YAW: <angle> into the serial monitor to check motor movement
   - If not already, the startup sequence should have moved the linear actuator to the bottom, and the user can type HEIGHT_MS: <ms> to manually check linear actuator movement.
   - The screen should display the default smiley face and the metrics bar should read PERCLOS: __ and posture: __
   - The speaker and wav file formats can be manually tested by typing WAV: <file name> into the serial monitor. Volume can be adjusted by typing VOL: <vol>.
   - If the IMU is enabled, it should read its values into the serial monitor.
6. Before running study_buddy.py, ensure that:
   - CAMERA_INDEX is set to the correct value. 0 is the default camera. Change to 1, 2, etc. to select different camera devices that are connected to the laptop. 
   - If the file is not running on Windows, change the frame capture and sound player to appropriate substitutes. 
   - The serial monitor on Arduino IDE is closed. Otherwise, the COM port will be busy
8. Running study_buddy.py:
   - Upon startup, the vision model will require a calibration, so the user should sit straight and comfortably for the first five seconds.
   - Another new calibration can manually be called by pressing R on the keyboard.
   - Take screenshots of the webcam feed and overlay by pressing S
   - Quit by pressing Q
