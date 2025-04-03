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
git clone https://github.com/sayakbiswas31/Crowd_Surveillance_ML_TEAM2.git
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

- # Image Detection System

## Overview
This project implements an image detection system using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. It processes images and classifies them into one of two categories using a trained deep learning model.

## Features
- Utilizes a pre-trained CNN model.
- Performs image classification with real-time or batch processing.
- Implements OpenCV for image handling.
- Supports GPU acceleration with TensorFlow.

## Project Structure
```
├── detectd.py        # Main script for image detection
├── model.json        # Trained Keras model in JSON format
├── README.md         # Project documentation
```

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install tensorflow opencv-python numpy
```

## How to Run
1. **Download the repository:**
   ```bash
   git clone  https://github.com/sayakbiswas31/Crowd_Surveillance_ML_TEAM2.git
   cd your-repo
   ```

2. **Ensure all dependencies are installed:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the detection script:**
   ```bash
   python detectd.py --image <path_to_image>
   ```
   - Replace `<path_to_image>` with the actual image file you want to process.

4. **View the output:**
   - The script will print the classification result in the terminal.
   - It may also display the processed image with bounding boxes (if applicable).

## Model Details
The CNN model consists of:
- Input layer for 250x250 RGB images.
- Multiple convolutional layers with ReLU activation.
- Max-pooling layers for feature extraction.
- Dense layers for classification.
- Softmax activation for binary classification.

The model is compiled with:
```python
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

## Contributions
Feel free to submit pull requests for improvements.

## License
This project is open-source under the MIT License.



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
git clone https://github.com/sayakbiswas31/Crowd_Surveillance_ML_TEAM2.git
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

- You can view the output of the Gender Classification  [here](https://drive.google.com/file/d/1DLjH7s4jnJyDD6HjMb612t_FAQPQwXJ4/view).

- # Trouble Detection Surveillance System

## Overview
This project implements a real-time trouble detection surveillance system using computer vision. It leverages **YOLOv8** for object detection and **DeepSORT** for multi-object tracking, allowing it to monitor and analyze crowd behavior in a given environment.

## Features
- **Person Detection**: Uses YOLOv8 to detect people in real-time.
- **Multi-Object Tracking**: Tracks individuals across frames using DeepSORT.
- **Proximity Analysis**: Measures distances between detected persons to identify potential trouble situations.
- **Live Video Processing**: Works with both live webcam feeds and pre-recorded video footage.

## Installation
### Requirements
Ensure you have Python installed. Then, install the required dependencies:
```bash
pip install ultralytics opencv-python numpy scipy deep-sort-realtime
```

## Usage
1. **Run the script:**
   ```bash
   python troubledetection_surveillance.py
   ```
2. **Provide a video source:**
   - Modify the script to use a webcam (`cv2.VideoCapture(0)`) or a video file (`cv2.VideoCapture('path/to/video.mp4')`).
3. **Analyze output:**
   - The script will detect and track individuals, highlighting their movements and possible trouble zones.

## How It Works
1. **YOLOv8 Person Detection:**
   - The YOLOv8 model (`yolov8x.pt`) detects people in each video frame.
2. **DeepSORT Multi-Object Tracking:**
   - Assigns unique IDs to each detected person, maintaining identity across frames.
3. **Proximity Analysis:**
   - Uses Euclidean distance to measure how close individuals are to each other.
   - Identifies potentially dangerous situations (e.g., fights, overcrowding).

## Customization
- Modify the `max_age` and `n_init` parameters in the `DeepSort` tracker to fine-tune tracking sensitivity.
- Adjust the detection confidence threshold in YOLO to reduce false positives.

## Future Enhancements
- Add action recognition models to detect specific types of trouble (e.g., violence detection).
- Implement alert systems to notify security personnel in real-time.
- Integrate with cloud-based monitoring solutions.

## Trouble Detection Output
You can view the output of the trouble detection system [here](https://drive.google.com/file/d/1uwXRH4n5qbNTnope7Zjc1goXvi0nn_Z0/view).



## License
This project is open-source under the **MIT License**.


Feel free to contribute by submitting **issues** or **pull requests**!

