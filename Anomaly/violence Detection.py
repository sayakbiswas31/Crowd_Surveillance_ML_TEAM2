import cv2
import torch
import mediapipe as mp
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use a custom-trained model for violence detection if available

# Initialize MediaPipe Pose Estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open video file
cap = cv2.VideoCapture('/Users/aaryangowda/Downloads/archive-2/Real Life Violence Dataset/Violence/V_301.mp4')

# Dictionary to store person IDs
person_tracker = {}
next_person_id = 1  # Start ID count from 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame)

    total_persons = 0
    weapon_detected = False
    updated_tracker = {}

    # Draw YOLO detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            label = model.names[cls]

            # Track persons with consistent IDs
            if label == "person":
                total_persons += 1
                found_existing_id = False

                # Try to find an existing person ID (based on proximity)
                for (prev_x1, prev_y1, prev_x2, prev_y2), pid in person_tracker.items():
                    if abs(prev_x1 - x1) < 50 and abs(prev_y1 - y1) < 50:
                        updated_tracker[(x1, y1, x2, y2)] = pid
                        found_existing_id = True
                        break

                # If no existing ID is found, assign a new one
                if not found_existing_id:
                    updated_tracker[(x1, y1, x2, y2)] = next_person_id
                    next_person_id += 1

                # Get assigned ID
                pid = updated_tracker[(x1, y1, x2, y2)]

                # Draw bounding box and ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Check for weapon detection
            elif label in ["knife", "gun", "weapon"]:  # Modify based on your model labels
                weapon_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Weapon Detected!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Update person tracker with new frame's IDs
    person_tracker = updated_tracker

    # Convert to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(rgb_frame)

    # Analyze keypoints for anomalies (Violent Behavior Detection)
    if results_pose.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        left_hand = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_hand = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        nose = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        left_elbow = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_knee = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        left_foot = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_foot = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

        if (left_hand.y < nose.y and right_hand.y < nose.y) or \
           (left_elbow.y < nose.y and right_elbow.y < nose.y) or \
           (abs(left_knee.y - left_foot.y) < 0.1 and abs(right_knee.y - right_foot.y) < 0.1):
            cv2.putText(frame, "Violence Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display total persons count
    cv2.putText(frame, f"Total Persons: {total_persons}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Anomaly Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
