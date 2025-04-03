# Crowd_Surveillance_ML_TEAM2

## Violence Detection using YOLOv8 and MediaPipe

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

- # Gender Classification using Vision Transformer (ViT) and YOLOv8

## Overview
This project implements a **gender classification system** using:
- **YOLOv8** for person detection
- **Haarcascade** for face detection
- **Vision Transformer (ViT)** for gender classification

It processes both **images and videos** to detect faces, extract features, and classify gender.

## Features
- Uses **YOLOv8** for detecting persons in images/videos.
- Applies **Haarcascade** for face detection.
- Employs **ViT (Vision Transformer)** for gender classification.
- Supports **image and video processing**.

## Installation
### Requirements
Ensure the following dependencies are installed:
```sh
pip install transformers torch torchvision pillow opencv-python ultralytics
```

### Clone Repository
```sh
git clone https://github.com/yourusername/gender-classification.git
cd gender-classification
```

## Usage
### Running on Images
Modify the script to load an image and run:
```sh
python gender_classification.py --image path_to_image.jpg
```

### Running on Videos
Modify the script to load a video and execute:
```sh
python gender_classification.py --video path_to_video.mp4
```

## Model Details
- Uses **YOLOv8n.pt** for person detection.
- Uses **ViT Large Patch16** for gender classification.
- Employs **Haarcascade** for face detection.

## Future Improvements
- Improving accuracy with **fine-tuned models**.
- Adding **real-time webcam support**.
- Enhancing **age classification** alongside gender detection.

## License
This project is open-source under the **MIT License**.


Feel free to contribute by submitting **issues** or **pull requests**!

