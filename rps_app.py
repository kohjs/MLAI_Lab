from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("my_model.hdf5")

# Use webcam only if running locally
USE_CAMERA = os.environ.get("USE_CAMERA", "1") == "1"
camera = None
if USE_CAMERA:
    camera = cv2.VideoCapture(0)

def preprocess_frame(frame):
    img = cv2.resize(frame, (75, 75))  # Resize to match training input
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # BGR to RGB
    img = np.asarray(img).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def get_prediction_label(pred):
    class_idx = np.argmax(pred)
    labels = ['Unknown', 'Pancake', 'Strawberry']
    return labels[class_idx]

def gen_frames():
    if not camera:
        raise RuntimeError("Camera is not available on this platform.")
    
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Preprocess and predict
        input_frame = preprocess_frame(frame)
        prediction = model.predict(input_frame)
        label = get_prediction_label(prediction)

        # Draw label on frame
        cv2.putText(frame, f'Detected: {label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode and yield frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if not USE_CAMERA:
        return "Video feed not supported on this platform."
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT environment variable
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
