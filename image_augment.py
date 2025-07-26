import os
import cv2
import albumentations as A

# Set input and output base paths
input_root = 'all_data'
output_root = 'augmented_data'
os.makedirs(output_root, exist_ok=True)

# Enhanced Albumentations pipeline for robust machine vision training
transform = A.Compose([
    # Geometric transforms
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=45, p=0.7),
    A.ShiftScaleRotate(
        shift_limit=0.15,
        scale_limit=0.2,
        rotate_limit=30,
        p=0.7
    ),
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    
    # Color transforms
    A.RandomBrightnessContrast(
        brightness_limit=0.3,
        contrast_limit=0.3,
        p=0.7
    ),
    A.HueSaturationValue(
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.5
    ),
    A.CLAHE(p=0.3),
    
    # Noise and blur
    A.GaussianBlur(blur_limit=(1, 3), p=0.4),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    
    # Quality transforms
    A.ImageCompression(quality_lower=75, quality_upper=95, p=0.3),
    A.RandomShadow(p=0.2),
    A.RandomSunFlare(p=0.1),
    
    # Regularization with CoarseDropout
    A.CoarseDropout(
        max_holes=8,
        max_height=16,
        max_width=16,
        min_holes=1,
        min_height=8,
        min_width=8,
        fill_value=0,
        mask_fill_value=None,
        p=0.3
    )
])

# Config
num_augmented = 10
max_images_per_class = 200

# Process each class
for class_name in os.listdir(input_root):
    class_input_dir = os.path.join(input_root, class_name)
    class_output_dir = os.path.join(output_root, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    # Get and sort up to 200 images
    image_files = sorted([
        f for f in os.listdir(class_input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])[:max_images_per_class]

    for image_name in image_files:
        image_path = os.path.join(class_input_dir, image_name)
        image = cv2.imread(image_path)

        # Generate augmentations (don't save original)
        for i in range(num_augmented):
            augmented = transform(image=image)['image']
            base_name = os.path.splitext(image_name)[0]
            aug_name = f"{base_name}aug{i}.jpg"
            aug_path = os.path.join(class_output_dir, aug_name)
            cv2.imwrite(aug_path, augmented)

print("✅ Done! Augmented images are saved in:",output_root)