import os
import cv2
import albumentations as A

input_dir = 'cropped_images'
output_dir = 'augmented_images'
os.makedirs(output_dir, exist_ok=True)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3)
])

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_dir, filename)
        image = cv2.imread(img_path)

        for i in range(5):
            augmented = transform(image=image)['image']
            cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"), augmented)