import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import warnings

# Suppress warnings related to the usage of torch.load (to avoid cluttering the console)
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.load.*")

# Define the CNN_LSTM model class
class CNN_LSTM(nn.Module):
    # The constructor to initialize the layers of the model
    def __init__(self, hidden_size=128, num_layers=1):
        super(CNN_LSTM, self).__init__()
        # Using MobileNetV2 as the CNN backbone
        self.cnn = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
        # Adaptive average pooling to reduce the feature map to a fixed size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # LSTM layer for processing sequences
        self.lstm = nn.LSTM(input_size=1280, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # Fully connected layer to output prediction
        self.fc = nn.Linear(hidden_size, 2)

    # The forward pass of the model
    def forward(self, x):
        B, T, C, H, W = x.size()  # B: Batch size, T: Time steps (frames), C: Channels, H: Height, W: Width
        cnn_features = []  # To store the CNN features for each time step (frame)
        
        for t in range(T):  # Loop through all frames
            # Get CNN features for each frame
            f = self.cnn(x[:, t])  # Apply CNN to each frame
            f = self.pool(f)  # Apply pooling to reduce dimensions
            f = f.view(B, -1)  # Flatten the features to a vector
            cnn_features.append(f)  # Store the features
        
        # Stack all features for the LSTM
        features = torch.stack(cnn_features, dim=1)  # Shape: (B, T, 1280)
        # Process the sequence of features through LSTM
        out, _ = self.lstm(features)
        # Get the final output of the LSTM (the last time step)
        out = self.fc(out[:, -1])  # Output layer
        return out  # Return the prediction

# Load the pre-trained model weights
model = CNN_LSTM()  # Create the model instance
# Load the weights from a file and map them to the CPU
model.load_state_dict(torch.load(r'C:\Users\Gopichand\OneDrive\Desktop\UPTOSKILLS\chain_snatching\snatching_model_71.43.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Parameters
SEQUENCE_LENGTH = 16  # The number of frames to process at once
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Choose device: GPU if available, else CPU
model = model.to(device)  # Move the model to the chosen device

# Open the video for detection
video_path = r'C:\Users\Gopichand\OneDrive\Desktop\UPTOSKILLS\chain_snatching\snatching_dataset\testing\snatching\snatching4.mp4'
cap = cv2.VideoCapture(video_path)  # Open the video file

frames = []  # List to store frames for the sequence
frame_count = 0  # Frame counter

# Process video frame by frame
while True:
    ret, frame = cap.read()  # Read a frame
    if not ret:  # If the frame was not read (i.e., end of video), break the loop
        break

    # Preprocess the frame
    frame_resized = cv2.resize(frame, (224, 224))  # Resize the frame to 224x224 pixels
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0  # Convert to tensor and normalize
    frames.append(frame_tensor)  # Add frame to the sequence

    frame_count += 1  # Increment the frame count

    # Once we have enough frames (a full sequence)
    if len(frames) == SEQUENCE_LENGTH:
        input_batch = torch.stack(frames).unsqueeze(0).to(device)  # Stack frames and add batch dimension

        with torch.no_grad():  # Disable gradient calculation for inference
            output = model(input_batch)  # Get the model's prediction
            prediction = torch.argmax(output, dim=1).item()  # Get the predicted class (0 or 1)

        # Only display the label if chain snatching is detected
        if prediction == 1:
            label = "Chain Snatching Detected!"  # Label to display on the frame
            color = (0, 0, 255)  # Red color for detected snatching
            print("Chain Snatching Detected!")  # Print message to the console
            # Display the label on the frame
            cv2.putText(frame_resized, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            # Show the frame with the label
            cv2.imshow('Detection', frame_resized)

        frames = []  # Clear frames for the next sequence

    else:
        # If not enough frames yet, just show the current frame without prediction
        cv2.imshow('Detection', frame_resized)

    # Check if the user pressed the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
