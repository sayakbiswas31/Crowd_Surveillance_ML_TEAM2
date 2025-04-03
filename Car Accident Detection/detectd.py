import cv2
from detection import AccidentDetectionModel
import numpy as np
import os

# Initialize the model
model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX


def startapplication():
    # Load the video (for camera use video = cv2.VideoCapture(0))
    video = cv2.VideoCapture('/Users/aaryangowda/Downloads/Screen Recording 2025-04-01 at 8.34.33 PM.mov')

    while True:
        # Read a frame from the video
        ret, frame = video.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame to RGB (if your model expects RGB input)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the frame to the input size expected by the model
        roi = cv2.resize(gray_frame, (250, 250))

        # Predict accident using the model
        pred, prob = model.predict_accident(roi[np.newaxis, :, :])

        if pred == "Accident":
            prob = (round(prob[0][0] * 100, 2))

            # Display the prediction and probability on the frame
            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"{pred} {prob}%", (20, 30), font, 1, (255, 255, 0), 2)

            # Optional: Beep when the probability is high
            # if prob > 90:
            #     os.system("say beep")

        # Display the frame
        cv2.imshow('Video', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    startapplication()