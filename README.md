---
title: AI Mask Surveillance System
emoji: 😷
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 😷 AI-Based Face Mask Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow)

## 🔴 Live Demo
👉 [Click here to try the app live](https://huggingface.co/spaces/Pratham4832/face-mask-detector)

---

## 📌 About the Project
A real-time face mask detection system built using Convolutional Neural Networks (CNN).
Upload any image and the system will automatically detect faces and classify
whether the person is wearing a mask or not — with confidence score.

---

## ✨ Features
- CNN-based face mask classifier
- Real-time face detection using Haar Cascade (OpenCV)
- Mask / No Mask classification with confidence score
- 96%+ classification accuracy
- Clean web interface built with Streamlit
- Deployed live using Docker on Hugging Face Spaces

---

## 🛠️ Tech Stack
| Technology | Purpose |
|---|---|
| Python | Core language |
| TensorFlow | Model training and inference |
| OpenCV | Face detection using Haar Cascade |
| Streamlit | Web application UI |
| Docker | Containerized deployment |
| Hugging Face Spaces | Free cloud deployment |

---

## 📁 Project Structure
```
AI-Mask-Surveillance-System/
├── app.py                    ← Streamlit web app
├── train_model.py            ← CNN model training script
├── detect_mask.py            ← Mask detection logic
├── detect_face.py            ← Face detection logic
├── preprocess_data.py        ← Data preprocessing
├── check_dataset.py          ← Dataset validation
├── mask_detector_model.keras ← Trained CNN model
├── requirements.txt          ← Python dependencies
├── Dockerfile                ← Docker deployment config
├── .streamlit/
│   └── config.toml           ← Streamlit configuration
└── README.md
```

---

## 🚀 How to Run Locally

**Step 1 — Clone the repository**
```bash
git clone https://github.com/Pratham-Barot/AI-Mask-Surveillance-System.git
cd AI-Mask-Surveillance-System
```

**Step 2 — Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**Step 3 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4 — Run the app**
```bash
streamlit run app.py
```

---

## 📊 Model Details
- Architecture: Convolutional Neural Network (CNN)
- Input size: 128x128x3
- Output: 2 classes (With Mask / Without Mask)
- Accuracy: 96%+
- Framework: TensorFlow + Keras

---

## 👨‍💻 Built By
**Pratham Barot**
- GitHub: [@Pratham-Barot](https://github.com/Pratham-Barot)
- Live Project: [Face Mask Detector](https://huggingface.co/spaces/Pratham4832/face-mask-detector)