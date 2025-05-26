# 📌 1. Install Ultralytics (YOLOv8)
!pip install ultralytics --quiet

# 📌 2. Import libraries
import cv2
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

# 📌 3. Load your YOLOv8 weapon detection model
model = YOLO('/content/best (12).pt')  # Uploaded model file

# 📌 4. Load your video
video_path = '/content/yes24.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# 📌 5. Prepare video writer to save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/content/output_detected.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

frame_count = 0

# 📌 6. Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run detection
    results = model(frame)
    annotated_frame = results[0].plot()

    # Show the frame in Colab
    cv2_imshow(annotated_frame)

    # Save the annotated frame to output video
    out.write(annotated_frame)

    print(f"Processed frame {frame_count}")

    # Optional: stop early (e.g., after 5 frames)
    #if frame_count >= 5:
        #break

cap.release()
out.release()