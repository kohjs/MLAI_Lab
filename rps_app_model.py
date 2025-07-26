from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import shutil
import random
import numpy as np

def create_random_split_directories(source_dirs, test_split=0.2, random_seed=None):
    """
    Creates temporary train/test directories with random split from multiple source directories
    """
    if random_seed is None:
        random_seed = random.randint(1, 10000)  # Different seed each time
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create temporary directories
    temp_train_dir = 'temp_train'
    temp_test_dir = 'temp_test'
    
    # Clean up existing temp directories
    if os.path.exists(temp_train_dir):
        shutil.rmtree(temp_train_dir)
    if os.path.exists(temp_test_dir):
        shutil.rmtree(temp_test_dir)
    
    os.makedirs(temp_train_dir)
    os.makedirs(temp_test_dir)
    
    print(f"Creating random split with {test_split*100}% test data...")
    
    # Process each source directory
    if isinstance(source_dirs, str):
        source_dirs = [source_dirs]
        
    # First pass: collect all class names
    all_classes = set()
    for source_dir in source_dirs:
        all_classes.update(os.listdir(source_dir))
    
    # Create class directories in temp folders
    for class_name in all_classes:
        os.makedirs(os.path.join(temp_train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(temp_test_dir, class_name), exist_ok=True)
    
    # Process each class directory from all sources
    for class_name in all_classes:
        all_image_files = []
        
        for source_dir in source_dirs:
            class_path = os.path.join(source_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            # Get all image files from this source
            image_files = [f for f in os.listdir(class_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            all_image_files.extend([(source_dir, f) for f in image_files])
        
        # Random split
        random.shuffle(all_image_files)
        split_idx = int(len(all_image_files) * (1 - test_split))
        train_files = all_image_files[:split_idx]
        test_files = all_image_files[split_idx:]
        
        print(f"Class '{class_name}': {len(train_files)} train, {len(test_files)} test images")
        
        # Copy files to temporary directories
        for source_dir, file in train_files:
            src = os.path.join(source_dir, class_name, file)
            dst = os.path.join(temp_train_dir, class_name, file)
            shutil.copy2(src, dst)
            
        for source_dir, file in test_files:
            src = os.path.join(source_dir, class_name, file)
            dst = os.path.join(temp_test_dir, class_name, file)
            shutil.copy2(src, dst)
    
    print(f"Random split created with seed: {random_seed}")
    return temp_train_dir, temp_test_dir, random_seed

def improved_image_gen_w_random_split(source_directories, test_split=0.2, random_seed=None):
    """
    Enhanced data augmentation with random train/test split from multiple sources
    """
    # Create random split directories
    temp_train_dir, temp_test_dir, used_seed = create_random_split_directories(
        source_directories, test_split, random_seed
    )
    
    train_datagen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=40,           # Increased rotation
        zoom_range=0.3,              # Increased zoom
        width_shift_range=0.2,       # Increased shift
        height_shift_range=0.2,
        shear_range=0.2,             # Added shear
        horizontal_flip=True,        # Added horizontal flip
        brightness_range=[0.8, 1.2], # Added brightness variation
        channel_shift_range=0.1,     # Added channel shift
        validation_split=0.25        # Larger validation split for small datasets
    )

    test_datagen = ImageDataGenerator(rescale=1/255)

    # Improved batch sizes for better gradient estimates
    train_generator = train_datagen.flow_from_directory(
        temp_train_dir,
        target_size=(150, 150),      # Increased input size for better features
        batch_size=32,               # Standard batch size
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        temp_train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        temp_test_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator, used_seed

def improved_model_output_for_TL(pre_trained_model, last_output, num_classes=3):
    """Improved model head with better architecture"""
    # Use GlobalAveragePooling instead of Flatten to reduce parameters
    x = GlobalAveragePooling2D()(last_output)
    
    # Add batch normalization for stability
    x = BatchNormalization()(x)
    
    # First dense layer with fewer neurons to reduce overfitting
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)  # Increased dropout
    
    # Second dense layer for better feature learning
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)  # Increased dropout
    
    # Output layer
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(pre_trained_model.input, predictions)
    return model

def create_callbacks():
    """Create useful callbacks for training"""
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,  # Reduced patience for faster stopping
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    return callbacks

def fine_tune_model(model, train_generator, validation_generator, unfreeze_layers=100):
    """Fine-tune the model by unfreezing some layers"""
    # Unfreeze the top layers of the pre-trained model
    for layer in model.layers[-unfreeze_layers:]:
        if not isinstance(layer, BatchNormalization):  # Keep BN layers frozen
            layer.trainable = True
    
    # Recompile with a lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=0.00005),  # Even lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def cleanup_temp_directories():
    """Clean up temporary directories after training"""
    temp_dirs = ['temp_train', 'temp_test']
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up {temp_dir}")

# Main training script with random splitting
def train_improved_model_with_random_split():
    # IMPORTANT: Change these to your dataset directories
    # These should contain subdirectories for each class with all your images
    source_dirs = [
        'C:/MLAI_Lab/all_data/',
        'C:/MLAI_Lab/augmented_data/'
    ]
    
    # Check if source directories exist
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"ERROR: Source directory '{source_dir}' does not exist!")
            print("Please update the source_dirs paths to point to your datasets.")
            print("Your directory structure should look like:")
            print("C:/MLAI_Lab/all_data/")
            print("├── pancake/")
            print("│   ├── pancake1.jpg")
            print("│   ├── pancake2.jpg")
            print("│   └── ...")
            print("├── strawberry/")
            print("│   ├── strawberry1.jpg")
            print("│   ├── strawberry2.jpg")
            print("│   └── ...")
            print("└── none/")
            print("    ├── other1.jpg")
            print("    ├── other2.jpg")
            print("    └── ... (images that are neither pancake nor strawberry)")
            print("\nAnd similarly for augmented_data/ directory")
            return None, None, None, None
    
    try:
        # Generate improved image data with random split
        train_generator, validation_generator, test_generator, used_seed = improved_image_gen_w_random_split(
            source_dirs,
            test_split=0.2,     # 20% for testing, 80% for training
            random_seed=None    # None = different split each time, or set a number for reproducible splits
        )

        # Load pre-trained InceptionV3 with improved input size
        pre_trained_model = InceptionV3(
            input_shape=(150, 150, 3),   # Larger input size
            include_top=False,
            weights='imagenet'
        )

        # Freeze all layers initially
        for layer in pre_trained_model.layers:
            layer.trainable = False

        # Use a later layer for better feature extraction
        last_layer = pre_trained_model.get_layer('mixed7')  # Deeper layer
        last_output = last_layer.output

        # Create improved model
        model_TL = improved_model_output_for_TL(pre_trained_model, last_output)

        # Compile with better optimizer settings
        model_TL.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Create callbacks
        callbacks = create_callbacks()

        # Calculate steps per epoch properly
        steps_per_epoch = train_generator.samples // train_generator.batch_size
        validation_steps = validation_generator.samples // validation_generator.batch_size

        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        print(f"Training with random seed: {used_seed}")

        # Phase 1: Train with frozen base
        print("Phase 1: Training with frozen base model...")
        history1 = model_TL.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=20,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        # Phase 2: Fine-tune with unfrozen layers
        print("Phase 2: Fine-tuning with unfrozen layers...")
        model_TL = fine_tune_model(model_TL, train_generator, validation_generator)
        
        # Reset callbacks for fine-tuning
        callbacks = create_callbacks()
        
        history2 = model_TL.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=15,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        # Combine histories
        history = {
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss']
        }

        return model_TL, history, test_generator, used_seed
    
    except Exception as e:
        print(f"Error during training: {e}")
        # Clean up in case of error
        cleanup_temp_directories()
        raise e

def plot_training_history(history):
    """Enhanced plotting function"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy plot
    axes[0, 0].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[0, 1].plot(history['loss'], label='Train Loss', linewidth=2)
    axes[0, 1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot (if available)
    axes[1, 0].set_title('Training Phases')
    axes[1, 0].text(0.1, 0.8, 'Phase 1: Frozen base (Epochs 1-20)', transform=axes[1, 0].transAxes)
    axes[1, 0].text(0.1, 0.6, 'Phase 2: Fine-tuning (Epochs 21-35)', transform=axes[1, 0].transAxes)
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    
    # Validation accuracy vs training accuracy difference
    val_train_diff = [val - train for val, train in zip(history['val_accuracy'], history['accuracy'])]
    axes[1, 1].plot(val_train_diff, linewidth=2, color='red')
    axes[1, 1].set_title('Validation - Training Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Difference')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_generator):
    """Comprehensive model evaluation"""
    print("Evaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    return test_accuracy, test_loss

# Run the improved training with random splitting
if __name__ == "__main__":
    print("Starting training with random train/test split...")
    
    try:
        # Train the model
        model, history, test_gen, seed = train_improved_model_with_random_split()
        
        if model is not None:
            # Plot training history
            plot_training_history(history)
            
            # Evaluate on test set
            evaluate_model(model, test_gen)
            
            # Save the final model
            model_filename = f'improved_inception_model_seed_{seed}.keras'
            model.save(model_filename)
            print(f"Model saved as '{model_filename}'")
            
            print(f"Training completed successfully with random seed: {seed}")
            print("Note: Each training run will use a different random split of your data.")
        
    except Exception as e:
        print(f"Training failed: {e}")
    
    finally:
        # Always clean up temporary directories
        cleanup_temp_directories()
        if model is not None:
            print("\nFinal Model Architecture:")
            model.summary()
        print("Training session ended.")