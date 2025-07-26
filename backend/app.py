from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from flask_cors import CORS
app = Flask(__name__)


CORS(app)
# Enhanced model loading with multiple fallbacks
model_paths = [
    'C:/MLAI_Lab/backend/model/improved_inception_model.keras',
]

model = None
for path in model_paths:
    try:
        print(f"Attempting to load model from: {path}")
        model = tf.keras.models.load_model(
            path,
            compile=False,
            custom_objects=None
        )
        print(f"Successfully loaded model from {path}")
        break
    except Exception as e:
        print(f"Failed to load {path}: {str(e)}")
        continue

if model is None:
    print("All model loading attempts failed. Please verify:")
    print("1. Model files exist in backend/model/")
    print("2. TensorFlow version is compatible")
    print("3. Model files are not corrupted")
    raise SystemExit(1)

# Load labels
labels = ['Unknown', 'Pancake', 'Strawberry']

# Set up camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Could not open camera. Make sure it's not being used by another application.")

def preprocess_frame(frame):
    img = cv2.resize(frame, (150, 150))
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = np.asarray(img).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def get_prediction_label(pred):
    class_idx = np.argmax(pred)
    return labels[class_idx]

def gen_frames():
    while True:
        try:
            success, frame = camera.read()
            if not success:
                print("Failed to read frame from camera")
                break
        except Exception as e:
            print(f"Camera error: {e}")
            break
        input_frame = preprocess_frame(frame)
        prediction = model.predict(input_frame)
        label = get_prediction_label(prediction)
        confidence = float(np.max(prediction))
        
        # Update latest prediction
        global latest_prediction
        latest_prediction = {
            "label": label,
            "confidence": confidence * 100
        }

        cv2.putText(frame, f'Detected: {label} ({confidence:.2f})', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Store the latest prediction
latest_prediction = {"label": "Unknown", "confidence": 0}

@app.route("/predict_image", methods=["POST"])
def predict_image():
    if 'image' not in request.files:
        return {"error": "No image provided"}, 400
    
    file = request.files['image']
    if file.filename == '':
        return {"error": "No selected file"}, 400
    
    try:
        # Process image similar to test.py
        img = Image.open(file.stream)
        img = img.resize((75, 75))
        img = np.asarray(img).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = model.predict(img)
        
        # Format response for frontend
        predictions = [
            {"label": "Unknown", "probability": float(prediction[0][0])},
            {"label": "Pancake", "probability": float(prediction[0][1])},
            {"label": "Strawberry", "probability": float(prediction[0][2])}
        ]
        
        return {"predictions": predictions}
    except Exception as e:
        return {"error": str(e)}, 500

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/latest_prediction")
def get_latest_prediction():
    return latest_prediction

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting server on port {port}")
    try:
        app.run(host="0.0.0.0", port=port, debug=True)
    except Exception as e:
        print(f"Server failed to start: {e}")
    finally:
        camera.release()
        print("Camera resources released")
