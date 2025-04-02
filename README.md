#Crowd_Surveillance_ML_TEAM2
##Violence Detection using YOLOv8 and MediaPipe

## Overview
This project implements a **violence detection system** using **YOLOv8** for object detection and **MediaPipe Pose** for human pose estimation. The script processes video input, detects persons, and analyzes their movements to identify potential violence.

## Features
- Uses **YOLOv8** for person and object detection.
- Employs **MediaPipe Pose Estimation** for analyzing human movements.
- Tracks multiple persons in a video frame.
- Designed for real-time violence detection.

## Installation
### Requirements
Ensure you have the following dependencies installed:

```sh
pip install opencv-python torch mediapipe ultralytics
```

### Clone Repository
```sh
git clone https://github.com/yourusername/violence-detection.git
cd violence-detection
```

## Usage
### Running the Script
```sh
python violence_detection.py
```
The script will process a sample video and attempt to detect violent activities.

### Input
Modify the `cap = cv2.VideoCapture('path_to_video.mp4')` line in the script to use your own video files.

### Output
- Frames with detected persons and their poses.
- Identifies potential violence using pose analysis.

## Dataset and Model
- Uses **YOLOv8n** pre-trained weights (`yolov8n.pt`).
- Compatible with custom-trained models for better accuracy.

## Future Improvements
- Integrating an **LSTM model** for time-series analysis.
- Improving classification accuracy with **custom-trained models**.
- Implementing **real-time webcam detection**.

## License
This project is open-source under the **MIT License**.



Feel free to contribute by submitting **issues** or **pull requests**!

