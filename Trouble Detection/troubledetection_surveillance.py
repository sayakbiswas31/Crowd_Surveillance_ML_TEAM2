import os
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial import distance  # To calculate how close people are

# Fix library conflicts (prevents some OpenCV errors)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the YOLOv8 model for person detection
model = YOLO('yolov8x.pt')

# Initialize the DeepSORT tracker (keeps track of people across frames)
tracker = DeepSort(max_age=10, n_init=3)  # Adjusted for smoother tracking

# Load the video file
video_path = r'C:\Users\Gopichand\OneDrive\Desktop\UPTOSKILLS\dataset_1\trouble.mp4'
cap = cv2.VideoCapture(video_path)

# Speed optimization (skip frames to process faster)
frame_skip = 3
frame_count = 0
prev_positions = {}

# Detection thresholds (adjust these to fine-tune alerts)
proximity_threshold = 80  # Distance (pixels) to consider people "close"
moderate_crowd_threshold = 8  # Medium-sized crowd
large_crowd_threshold = 16  # Large crowd alert
explosion_brightness_threshold = 150  # Explosion detection (brightness level)
panic_speed_threshold = 25  # Speed (pixels per frame) to consider "panic"
panic_agreement_threshold = 60  # % of people moving in the same direction
panic_min_people = 4  # Minimum number of people required for panic detection

# Panic detection cooldown (prevents rapid false alarms)
panic_cooldown_frames = 50  
panic_cooldown_counter = 0  
panic_active = False  # Tracks whether panic is currently active

# Store movement history (used to smooth out direction tracking)
prev_directions = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if no frame is read

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip frames for faster processing

    # Convert frame to grayscale for explosion detection (brightness check)
    gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_curr)

    # Run YOLO person detection (lower confidence for better detection)
    results = model(frame, conf=0.15, agnostic_nms=True, verbose=False)

    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            if cls == 0:  # Class 0 = person
                detections.append(([x1, y1, x2, y2], conf, cls))

    # Use DeepSORT to track detected people
    tracks = tracker.update_tracks(detections, frame=frame)

    # Store tracked people positions
    tracked_positions = {}
    for track in tracks:
        if track.is_confirmed() and track.time_since_update == 0:
            x1, y1, x2, y2 = track.to_tlbr()
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            tracked_positions[track.track_id] = (center_x, center_y)

    # Check how many people are close together (crowd detection)
    close_pairs = 0
    if len(tracked_positions) > 1:
        coords = list(tracked_positions.values())
        dists = distance.cdist(coords, coords, 'euclidean')
        close_pairs = np.sum(dists < proximity_threshold) // 2

    # Generate alert messages based on crowd size
    alert_messages = []

    if close_pairs >= large_crowd_threshold:
        alert_messages.append("🚨 Large Crowd Detected!")
    elif close_pairs >= moderate_crowd_threshold:
        alert_messages.append("⚠️ Moderate Crowd Detected!")

    # Explosion detection (based on frame brightness)
    explosion_detected = brightness > explosion_brightness_threshold
    if explosion_detected:
        alert_messages.append("🔥 Explosion/Fire Detected!")

    # Panic detection (check movement speed & direction)
    movement_speeds = []
    movement_directions = []

    if prev_positions:
        for track_id, (x, y) in tracked_positions.items():
            if track_id in prev_positions:
                prev_x, prev_y = prev_positions[track_id]
                movement_x = x - prev_x
                movement_y = y - prev_y
                speed = np.sqrt(movement_x**2 + movement_y**2)

                if speed > 5:  # Ignore small movements
                    movement_speeds.append(speed)
                    movement_directions.append((movement_x, movement_y))

    prev_positions = tracked_positions.copy()

    # Smooth movement direction using rolling average
    if movement_directions:
        prev_directions.append(np.mean(movement_directions, axis=0))
        if len(prev_directions) > 5:
            prev_directions.pop(0)  # Keep last 5 frames

    # Panic Detection Logic
    panic_detected = False
    if len(movement_speeds) >= panic_min_people:
        fast_movers = np.sum(np.array(movement_speeds) > panic_speed_threshold)
        fast_movers_percentage = (fast_movers / len(movement_speeds)) * 100

        if prev_directions:
            avg_direction = np.mean(prev_directions, axis=0)
            same_direction_count = np.sum(
                np.linalg.norm(np.array(movement_directions) - avg_direction, axis=1) < 10
            )
            panic_percentage = (same_direction_count / len(movement_directions)) * 100
        else:
            panic_percentage = 0

        # Ensure panic alert isn't triggered repeatedly
        if panic_percentage > panic_agreement_threshold and fast_movers_percentage > 50:
            if not panic_active and panic_cooldown_counter == 0:
                panic_detected = True
                panic_active = True  
                panic_cooldown_counter = panic_cooldown_frames  

    # If explosion is detected, automatically trigger panic
    if explosion_detected:
        panic_detected = True
        alert_messages.append("🚨 PANIC DETECTED! 🚨")

    # Manage cooldown (prevents rapid on/off switching of panic alert)
    if panic_cooldown_counter > 0:
        panic_cooldown_counter -= 1
    else:
        panic_active = False  

    if panic_detected:
        alert_messages.append("🚨 Panic Detected!")

    # Print alerts in console
    for alert in alert_messages:
        print(alert)

    # Draw bounding boxes around detected people
    for (x1, y1, x2, y2), conf, cls in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display alert messages on the video
    y_offset = 30
    for alert in alert_messages:
        cv2.putText(frame, str(alert), (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y_offset += 40

    # Show the video feed
    cv2.imshow('Crowd Surveillance', frame)

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and close video
cap.release()
cv2.destroyAllWindows()
