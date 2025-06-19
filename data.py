import os
import shutil
from sklearn.model_selection import train_test_split
original_dataset_dir = 'C:/Users/agraw/OneDrive - Manipal University Jaipur/Desktop/CVPR PROJECT'
base_dir = 'vehicle_data_split'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
val_ratio = 0.2

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)


for class_name in os.listdir(original_dataset_dir):
    class_path = os.path.join(original_dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [img for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(images) == 0:
        print(f"⚠️ Skipping empty class folder: {class_name}")
        continue

    train_imgs, val_imgs = train_test_split(images, test_size=val_ratio, random_state=42)

    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
    for img in val_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))
