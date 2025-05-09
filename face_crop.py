import os
from facenet_pytorch import MTCNN
from PIL import Image

input_dir = 'images'
output_dir = 'cropped_images'
os.makedirs(output_dir, exist_ok=True)

# Detect only one face (or change to keep_all=True if multiple faces)
mtcnn = MTCNN(keep_all=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert('RGB')  # Ensure correct mode
        boxes, _ = mtcnn.detect(img)

        if boxes is not None:
            for i, box in enumerate(boxes):
                face = img.crop(box)
                face.save(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_face_{i}.jpg"))
        else:
            print(f"No face detected in: {filename}")
