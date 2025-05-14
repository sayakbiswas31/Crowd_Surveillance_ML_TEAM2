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

- **Accuracy**: ✅ **79%**


#  Accident Detection using Custom CNN

This project implements an **accident detection system** using a Convolutional Neural Network (CNN) trained with Keras. The system analyzes images or frames (not real-time video) and classifies whether they indicate an accident or not.

---

##  Model Overview (`model.json`)

The CNN model is structured to classify an input image as either:

- **0** – No Accident  
- **1** – Accident Detected

### 🔍 Architecture:

- **Input Shape**: `(250, 250, 3)`
- **Layers**:
  - `BatchNormalization`
  - `Conv2D` layers (32, 64, 128, 256 filters)
  - `MaxPooling2D` layers after each Conv2D
  - `Flatten` layer to flatten feature maps
  - `Dense(512)` with ReLU
  - Final `Dense(2)` with Softmax for binary output
- **Loss Function**: `sparse_categorical_crossentropy`
- **Accuracy**: ✅ **89%**

---

## 🧪 detection.py – Image Classification

This script provides the core functionality for:

- Loading the model architecture and weights
- Preprocessing input images
- Predicting whether an accident is detected
- Outputting the result as a label (`Accident` or `No Accident`)

### Sample usage:
```python
from detection import detect

result = detect("path_to_image.jpg")
print("Prediction:", result)
```

---

## 🛠️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/sayakbiswas31/Crowd_Surveillance_ML_TEAM2.git
cd Crowd_Surveillance_ML_TEAM2
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt**
```txt
tensorflow
opencv-python
numpy
```

### 3. Run detection on an image
Make sure your image is in the correct format, then run:
```bash
python detection.py --image path_to_image.jpg
```

> (Optional: You can modify `detection.py` to support CLI input.)

---

## 📂 Project Structure

```
Crowd_Surveillance_ML_TEAM2/
├── detection.py        # Contains prediction code
├── model.json          # CNN model architecture
├── model_weights.h5    # Pre-trained weights (add manually)
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

---

## 🧾 Output Example

After running `detection.py`, you'll get output like:
```
Prediction: Accident
```
or
```
Prediction: No Accident
```

---

## ⚠️ Notes

- Make sure `model_weights.h5` is present in the same directory.
- Input images must match the model input size (250×250), or be resized appropriately.
- Model accuracy: **89%**



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

- - **Accuracy**: ✅ **85%**

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

- **Accuracy**: ✅ **83%**

- 
## License
This project is open-source under the **MIT License**.


Feel free to contribute by submitting **issues** or **pull requests**!


- # Chain Snatching Detection System

## Overview
This project implements a real-time chain snatching detection system using deep learning. It leverages a CNN-LSTM architecture to identify suspicious events.

## Features
- CNN + LSTM Detection: Uses MobileNetV2 and LSTM to detect actions over time.
- Sequence-Based Analysis: Processes 16-frame sequences.
- Real-Time Video Inference: Supports live and pre-recorded videos.
- Visual and Console Alerts: Displays alerts on screen and prints in terminal.

## Installation
- Install required Python libraries: pip install torch torchvision opencv-python

## Usage
1. Update file paths: Modify video path and model path.
2. Run the script: python chain_snatching_detection.py

## How It Works
1. CNN Feature Extraction: Extracts visual features using MobileNetV2.
2. LSTM Temporal Modeling: Analyzes sequences of frames.
3. Binary Classification: Predicts chain snatching events.

## Accuracy
- 81.43%

## License
- Open-source under the MIT License.




- # Body Classification 
## Overview
This project implements a body-based gender classification system using YOLOv5 for person detection and ResNet18 for gender classification. It processes videos, extracts human figures, and classifies them as male or female based on full-body appearance.

## Features
Detects persons in videos using YOLOv5.

Extracts and preprocesses body crops from video frames.

Trains a ResNet18 CNN to classify gender (male/female).

Supports real-time gender classification in video streams.

Annotates frames with bounding boxes and gender labels.

## Installation
## Requirements
Install the required dependencies:

pip install torch torchvision albumentations opencv-python numpy pillow matplotlib rembg
## Dataset and Preprocessing
Step 1: Split Dataset
Organize your dataset into two folders: male/ and female/. Then split into training and validation sets.
split_dataset('/content/drive/MyDrive/dataset', '/content/drive/MyDrive/dataset_split', train_ratio=0.8)

Step 2: Extract Frames from Videos
Extract frames from each video to use for person detection:
extract_frames_from_videos(video_dir, output_dir, frame_skip=10)

Step 3: Crop Persons from Frames
Use YOLOv5 to detect and crop person bounding boxes from frames:
crop_person_from_images(input_dir, output_dir)
Repeat the process for both train and val splits for each class (male, female).

## Model Training
## Model
Architecture: ResNet18 (pretrained on ImageNet)

## Input Size: 224x224

## Output Classes: Male, Female

## Training Code

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
Use standard data augmentation and normalization:

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
## Train the model for 10 epochs:

for epoch in range(10):
    model.train()
    ...
    print(f"Epoch {epoch+1}, Loss: {running_loss}")
## Save the trained model:

torch.save(model.state_dict(), '/content/model.pth')
## Inference on Video
## Process
Load YOLOv5 to detect persons in each frame.
Crop persons and classify gender with ResNet18.
Annotate the video with predictions.

## Sample Code

# Detect persons
results = yolo_model(temp_img_path)
persons = results.pandas().xyxy[0][results.pandas().xyxy[0]['name'] == 'person']

# Crop and classify each person
for index, row in persons.iterrows():
    crop = frame[y1:y2, x1:x2]
    ...
    output = model(input_tensor)
    label = class_names[pred.item()]
    ...
## Output
Annotated video frames with bounding boxes.

Gender labels printed per frame:

Frame 5: ['male', 'female', 'male']
## Usage
Running the Pipeline
python body_gender_classification.py

## Modify the script to point to your video:
cap = cv2.VideoCapture('/content/drive/MyDrive/crowd.mp4')

## Future Improvements
Integrate DeepSORT for person tracking across frames.
Add pose estimation to support more nuanced body analysis.
Train with larger and more diverse datasets.
Extend to age group or attire classification.

## Acknowledgments
Ultralytics YOLOv5
TorchVision Models
