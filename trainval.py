import os
import shutil
from sklearn.model_selection import train_test_split


image_dir = 'data_object_image_2/training/image_2/'
label_dir = 'yolo_labels/'

output_image_train = 'dataset/images/train/'
output_image_val = 'dataset/images/val/'
output_label_train = 'dataset/labels/train/'
output_label_val = 'dataset/labels/val/'

os.makedirs(output_image_train, exist_ok=True)
os.makedirs(output_image_val, exist_ok=True)
os.makedirs(output_label_train, exist_ok=True)
os.makedirs(output_label_val, exist_ok=True)

image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

train_images, val_images = train_test_split(image_files, test_size=0.2, random_state=42)

def move_files(image_list, image_output_dir, label_output_dir):
    for image_file in image_list:

        shutil.copy(os.path.join(image_dir, image_file), image_output_dir)

        label_file = image_file.replace('.png', '.txt')
        shutil.copy(os.path.join(label_dir, label_file), label_output_dir)

move_files(train_images, output_image_train, output_label_train)
move_files(val_images, output_image_val, output_label_val)
