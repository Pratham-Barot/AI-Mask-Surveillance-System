import cv2
import numpy as np
import tensorflow as tf
import time
import os
from datetime import datetime
import winsound  # Windows only

# -----------------------------
# CONFIGURATION
# -----------------------------
CONFIDENCE_THRESHOLD = 0.80
SAVE_DELAY = 3  # seconds between saving violations

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model("mask_detector_model.h5")
print("Model loaded successfully.")

# -----------------------------
# Load Face Detector
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Create Folders
# -----------------------------
if not os.path.exists("violations"):
    os.makedirs("violations")

# Create log file if not exists
if not os.path.exists("log.txt"):
    with open("log.txt", "w") as f:
        f.write("Timestamp | Status | Confidence\n")

# -----------------------------
# Start Webcam
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

print("Press 'q' to exit")

# -----------------------------
# Counters
# -----------------------------
total_faces = 0
mask_count = 0
violation_count = 0
last_save_time = 0
prev_time = 0

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        total_faces += 1
        face = frame[y:y+h, x:x+w]

        # Preprocess face
        face_resized = cv2.resize(face, (128, 128))
        face_normalized = face_resized / 255.0
        face_reshaped = np.reshape(face_normalized, (1, 128, 128, 3))

        prediction = model.predict(face_reshaped, verbose=0)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        if confidence >= CONFIDENCE_THRESHOLD:

            if class_index == 0:
                label = f"With Mask ({confidence*100:.2f}%)"
                color = (0, 255, 0)
                mask_count += 1

            else:
                label = f"Without Mask ({confidence*100:.2f}%)"
                color = (0, 0, 255)

                current_time = time.time()

                # Smart Saving (delay)
                if current_time - last_save_time > SAVE_DELAY:
                    violation_count += 1
                    last_save_time = current_time

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_path = f"violations/violation_{timestamp}.jpg"

                    # Save cropped face
                    cv2.imwrite(file_path, face)

                    # Beep alert
                    winsound.Beep(1000, 300)

                    # Logging
                    with open("log.txt", "a") as f:
                        f.write(
                            f"{datetime.now()} | Without Mask | {confidence*100:.2f}%\n"
                        )

        else:
            label = "Low Confidence"
            color = (255, 255, 0)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # -----------------------------
    # FPS Calculation
    # -----------------------------
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # -----------------------------
    # Timestamp Overlay
    # -----------------------------
    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    cv2.putText(frame, now, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # -----------------------------
    # Counters Display
    # -----------------------------
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.putText(frame, f"Total Faces: {total_faces}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"With Mask: {mask_count}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"Violations: {violation_count}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("AI Mask Surveillance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()