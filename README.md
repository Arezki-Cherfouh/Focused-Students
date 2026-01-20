Student Focus Monitor

A Python application that uses MediaPipe and OpenCV to monitor student engagement in real-time.

Features

Person Detection: Detects people using MediaPipe Pose estimation
Focus Detection: Analyzes head position and posture to determine if student is focused
Green rectangle: Student is focused (head upright, facing forward)
Orange rectangle: Student is distracted (looking away or down)
Hand Raise Detection: Detects when a student raises their hand
Displays "RAISED HAND" text above the bounding box
Shows a red indicator dot

Installation

1. Install the required dependencies:
   bash
   pip install -r requirements.txt

Usage

Run the script:
bash
python main.py

Press 'q' to quit the application.

How It Works

Focus Detection
The system determines focus based on:
Head orientation: Checks if the student is looking forward (not turned sideways)
Head position: Ensures the head is upright (not looking down)

Hand Raise Detection
Detects raised hands by checking if either wrist is:
Above the shoulder level
Near or above head level
Clearly visible to the camera

Visual Indicators
Green box + "Focused": Student is paying attention
Orange box + "Distracted": Student appears distracted
Red text "RAISED HAND": Student has raised their hand

Requirements

Python 3.7+
Webcam
Good lighting conditions for best detection accuracy

Tips for Best Results

1. Ensure students are well-lit and clearly visible
2. Camera should be positioned to capture upper body and head
3. Minimize background clutter for better detection
4. Students should be 2-6 feet from the camera

Customization

You can adjust detection sensitivity by modifying these parameters in the code:
`min_detection_confidence`: Default 0.5 (line 9)
`min_tracking_confidence`: Default 0.5 (line 10)
Head distance threshold for focus detection (line 61)
Hand raise height threshold (line 41-42)
