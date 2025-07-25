from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def image_gen_w_aug(train_parent_directory, test_parent_directory):
    # Define the training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.15
    )

    # Define the test data generator (no augmentation, only rescaling)
    test_datagen = ImageDataGenerator(rescale=1/255)

    # Training data generator (85% of training data)
    train_generator = train_datagen.flow_from_directory(
        train_parent_directory,
        target_size=(75, 75),
        batch_size=214,
        class_mode='categorical',
        subset='training'
    )

    # Validation data generator (15% of training data)
    val_generator = train_datagen.flow_from_directory(
        train_parent_directory,
        target_size=(75, 75),
        batch_size=37,
        class_mode='categorical',
        subset='validation'
    )

    # Test data generator (from a separate directory)
    test_generator = test_datagen.flow_from_directory(
        test_parent_directory,
        target_size=(75, 75),
        batch_size=37,
        class_mode='categorical'
    )

    return train_generator, val_generator, test_generator

def model_output_for_TL(pre_trained_model, last_output):
    x = Flatten()(last_output)
    # Dense hidden layer
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    # Output layer for 3 classes
    x = Dense(3, activation='softmax')(x)
    
    model = Model(pre_trained_model.input, x)
    return model

# Set up directory paths
train_dir = os.path.join('C:/MLAI/MLAI_Lab/train/')
test_dir = os.path.join('C:/MLAI/MLAI_Lab/test/')

# Generate image data with augmentation
train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, test_dir)

# Load pre-trained InceptionV3 without the top layer
pre_trained_model = InceptionV3(
    input_shape=(75, 75, 3),
    include_top=False,
    weights='imagenet'
)
# Freeze all layers in the pre-trained model
for layer in pre_trained_model.layers:
    layer.trainable = False  # frozen layers won’t be updated during training

# Define the output from a specific intermediate layer
last_layer = pre_trained_model.get_layer('mixed3')
last_output = last_layer.output

# Create the full model using the custom output head
model_TL = model_output_for_TL(pre_trained_model, last_output)

# Compile the model
model_TL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history_TL = model_TL.fit(
    train_generator,
    steps_per_epoch=10,
    epochs=15,
    verbose=1,
    validation_data=validation_generator
)

# Plot Accuracy
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history_TL.history['accuracy'], label='Train Accuracy')
plt.plot(history_TL.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history_TL.history['loss'], label='Train Loss')
plt.plot(history_TL.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
# Save the trained model to file
save_path = os.path.join(os.path.dirname(__file__), 'my_model.hdf5')
tf.keras.models.save_model(model_TL, save_path)