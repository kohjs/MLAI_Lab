import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

def load_and_prepare_image(img_path, target_size=(150, 150)):
    """Load and preprocess image for model prediction"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def test_single_image(model_path, image_path, class_names=['none', 'pancake', 'strawberry']):
    """Test a single image and display detailed results"""
    try:
        # Load trained model
        model = tf.keras.models.load_model(model_path)
        
        # Prepare image
        img_array = load_and_prepare_image(image_path)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
        
        # Display image
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        img = image.load_img(image_path)
        plt.imshow(img)
        plt.title(f"Input Image\n{os.path.basename(image_path)}")
        plt.axis('off')
        
        # Display results as text
        plt.subplot(1, 2, 2)
        plt.axis('off')
        result_text = "Prediction Results:\n\n"
        for name, prob in zip(class_names, predictions[0]):
            result_text += f"{name}: {prob:.6f} ({prob*100:.2f}%)\n"
            if name == predicted_class:
                result_text += "  ^ Predicted\n"
        
        plt.text(0.1, 0.5, result_text, 
                fontfamily='monospace',
                verticalalignment='center')
        plt.title(f"Final Prediction: {predicted_class}\nConfidence: {confidence:.2f}%")
        plt.tight_layout()
        plt.show()
        
        # Print detailed info to console
        print("\n=== Prediction Details ===")
        print(f"Image: {image_path}")
        print(f"Model: {model_path}")
        print("\nPrediction Results:")
        for name, prob in zip(class_names, predictions[0]):
            print(f"{name}:")
            print(f"  Raw value: {prob:.6f}")
            print(f"  Percentage: {prob*100:.2f}%")
            if name == predicted_class:
                print("  ^ Predicted")
        print(f"\nFinal Prediction: {predicted_class} ({confidence:.2f}% confidence)")
        
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    MODEL_PATH = "backend/model/improved_inception_model.keras"
    IMAGE_PATH = "C:\MLAI_Lab\stockimages/360_F_415877698_c1VY6BMnUbNxh7Or80VDZAWng4UoGfWi.jpg"
    
    print("Starting image tester...")
    test_single_image(MODEL_PATH, IMAGE_PATH)