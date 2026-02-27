import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="AI Mask Detection", layout="centered")

st.title("😷 AI-Based Face Mask Detection System")
st.write("Upload an image to detect Mask / No Mask")

# ----------------------------
# Load Model (Cached)
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mask_detector_model.h5")

model = load_model()

# ----------------------------
# Load Haar Cascade
# ----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("⚠ No face detected in the image.")
    else:
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            # Preprocess
            face_resized = cv2.resize(face, (128, 128))
            face_normalized = face_resized / 255.0
            face_reshaped = np.reshape(face_normalized, (1, 128, 128, 3))

            prediction = model.predict(face_reshaped, verbose=0)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            if class_index == 0:
                label = "😷 With Mask"
                color = (0, 255, 0)
                st.success(f"Prediction: WITH MASK ({confidence:.2f}%)")
            else:
                label = "❌ Without Mask"
                color = (0, 0, 255)
                st.error(f"Prediction: WITHOUT MASK ({confidence:.2f}%)")

            # Draw on image
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{confidence:.2f}%",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Detection Result", use_container_width=True)

st.markdown("---")
st.caption("Built by Pratham | AI/ML Project")