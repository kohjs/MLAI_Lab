import os
import cv2
import albumentations as A

# Set input and output base paths
input_root = 'all_data'
output_root = 'augmented_data'
os.makedirs(output_root, exist_ok=True)

# Albumentations pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.3),
])

# Config
num_augmented = 5
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

        # Save original to output
        cv2.imwrite(os.path.join(class_output_dir, image_name), image)

        # Generate augmentations
        for i in range(num_augmented):
            augmented = transform(image=image)['image']
            base_name = os.path.splitext(image_name)[0]
            aug_name = f"{base_name}aug{i}.jpg"
            aug_path = os.path.join(class_output_dir, aug_name)
            cv2.imwrite(aug_path, augmented)

print("✅ Done! Augmented images are saved in:",output_root)