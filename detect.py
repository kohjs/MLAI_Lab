from PIL import Image, ImageOps
import tensorflow as tf
import cv2
import numpy as np
import os
import sys

label = ''
frame = None

# Function to process and predict from image
def import_and_predict(image_data, model):
    size = (150, 150)  # set the image size to match the model input
    image = ImageOps.fit(image_data, size, method=Image.Resampling.LANCZOS)  # resize with anti-aliasing
    image = image.convert('RGB')  # convert image to RGB
    image = np.asarray(image)  # convert image into array
    image = (image.astype(np.float32) / 255.0)  # normalize the image
    img_reshape = image[np.newaxis, ...]  # add batch dimension
    prediction = model.predict(img_reshape)  # make prediction
    return prediction

# Load model
model = tf.keras.models.load_model('C:/MLAI/MLAI_Lab/improved_inception_model.keras')

# Start webcam
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Camera OK")
else:
    cap.open()

while True:
    ret, original = cap.read()
    if not ret:
        continue

    # Resize frame for both display and prediction
    resized_frame = cv2.resize(original, (150, 150))  # Resize to the required 150x150
    cv2.imwrite(filename='img.png', img=resized_frame)  # Save resized image for prediction

    image = Image.open('img.png')
    prediction = import_and_predict(image, model)

    # Interpret prediction
    if np.argmax(prediction) == 0:
        predict = "It is unknown!"
    elif np.argmax(prediction) == 1:
        predict = "It is a pancake!"
    else:
        predict = "It is a strawberry!"

    # Display prediction on frame
    cv2.putText(original, predict, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", original)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sys.exit()

