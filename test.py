import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from PIL import Image, ImageOps

# Function to process and predict image
def import_and_predict(image_data, model):
    size = (75, 75)  # set the image size
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)  # resize with anti-aliasing
    image = image.convert('RGB')  # convert image to RGB format
    image = np.asarray(image)  # convert image into array
    image = (image.astype(np.float32) / 255.0)  # normalize pixel values
    img_reshape = image[np.newaxis, ...]  # add batch dimension
    prediction = model.predict(img_reshape)  # make prediction
    return prediction  # return predicted output

# Load the trained model
model = tf.keras.models.load_model('C:/MLAI_Lab/backend/model/improved_inception_model.keras')

# Load pre-trained InceptionV3 just for structure if needed (not required for prediction)
pre_trained_model = InceptionV3(
    input_shape=(75, 75, 3),
    include_top=False,
    weights='imagenet'
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()

    # Load and predict on the image
    image = Image.open(args.image)
    prediction = import_and_predict(image, model)
    
    # Print predictions as JSON array
    print([float(x) for x in prediction[0]])
st.write("""
# Strawberry and Pancake Prediction
""")

st.write("This is a simple image classification web app to detect strawberries and pancakes.")

# Upload image
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("You haven't uploaded an image file.")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    prediction = import_and_predict(image, model)

    # Show result
    if np.argmax(prediction) == 0:
        st.write("It is a **Unknown**!")
    elif np.argmax(prediction) == 1:
        st.write("It is a **Pancake**!")
    else:
        st.write("It is a **Strawberry**!")

    st.text("Prediction Probabilities (0: Unknown, 1: Pancake, 2: Strawberry):")
    st.write(prediction)
