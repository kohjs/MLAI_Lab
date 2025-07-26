from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from flask_cors import CORS
app = Flask(__name__)


CORS(app)
# Load model
import os
model_path = os.path.join(os.path.dirname(__file__), "model", "improved_inception_model.keras")
model = tf.keras.models.load_model(model_path)

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
