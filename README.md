# Face Emotion Recognition App 🎭

A web-based **Face Emotion Recognition (FER)** application that detects human facial emotions in real-time using **Haar Cascade** for face detection and a **Convolutional Neural Network (CNN)** model trained on the **FER-2013 dataset**.

---

## 📌 Features
- Real-time face detection using **OpenCV Haar Cascade**
- Emotion classification using **CNN (Keras / TensorFlow)**
- Displays:
  - Detected face
  - Predicted emotion
  - Confidence level
- Web interface built with **Flask**
- Supports live webcam input

---

## 😃 Detected Emotions
The model is trained to classify the following emotions:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Neutral
- Surprise
> ⚠️ Note: The model still has difficulty recognizing the **Surprise** emotion accurately.

---

## 🧠 Model Information
- Dataset: **FER-2013**
- Model type: **Convolutional Neural Network (CNN)**
- Framework: **TensorFlow / Keras**
- Model format: `.keras`
- Input: Grayscale face image (48x48)
- Output: Emotion label + confidence score
- Accuracy: ~ **65%**

---

## 🛠️ Tech Stack
- Python 3.9+
- TensorFlow / Keras
- OpenCV
- Flask
- NumPy
- HTML / CSS

---

## 🚀 How to Run Locally

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/face_emotion_recog_apps.git
cd face_emotion_recog_apps

python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

pip install -r requirements.txt

python app.py

http://127.0.0.1:5000


