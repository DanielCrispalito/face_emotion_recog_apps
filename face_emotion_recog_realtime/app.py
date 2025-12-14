from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ===== PATH AMAN =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== LOAD MODEL =====
model = load_model(os.path.join(BASE_DIR, "best_fer2013_model.keras"))

# ===== LOAD HAAR CASCADE =====
face_cascade = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
)

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']


# ===== ROUTES =====
@app.route("/")
def index():
    return render_template("dashboard.html", labels=emotion_labels)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    img_bytes = np.frombuffer(file.read(), np.uint8)

    # Decode image
    frame = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    # Face detection
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    if len(faces) == 0:
        return jsonify({"error": "No face detected"})

    x, y, w, h = faces[0]
    roi = frame[y:y+h, x:x+w]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float32") / 255.0
    roi = np.reshape(roi, (1, 48, 48, 1))

    preds = model.predict(roi, verbose=0)[0]

    return jsonify({
        "emotion": emotion_labels[int(np.argmax(preds))],
        "confidence": float(np.max(preds)),
        "scores": preds.tolist()
    })


if __name__ == "__main__":
    app.run()
