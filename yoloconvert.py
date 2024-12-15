import os
import cv2


kitti_labels_dir = 'data_object_label_2/training/label_2/'
kitti_images_dir = 'data_object_image_2/training/image_2/'
yolo_labels_dir = 'yolo_labels/'

os.makedirs(yolo_labels_dir, exist_ok=True)

class_mapping = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

def convert_kitti_to_yolo(label_file, image_file, output_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()

    img = cv2.imread(image_file)
    height, width, _ = img.shape

    yolo_labels = []
    for line in lines:
        parts = line.strip().split(' ')
        class_name = parts[0]

        if class_name in class_mapping:
            class_id = class_mapping[class_name]

            bbox_left = float(parts[4])
            bbox_top = float(parts[5])
            bbox_right = float(parts[6])
            bbox_bottom = float(parts[7])

            x_center = (bbox_left + bbox_right) / 2.0 / width
            y_center = (bbox_top + bbox_bottom) / 2.0 / height
            bbox_width = (bbox_right - bbox_left) / width
            bbox_height = (bbox_bottom - bbox_top) / height

            yolo_labels.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}")

    with open(output_file, 'w') as f:
        for label in yolo_labels:
            f.write(f"{label}\n")

for label_file in os.listdir(kitti_labels_dir):
    if label_file.endswith(".txt"):

        img_file = os.path.join(kitti_images_dir, label_file.replace(".txt", ".png"))
        yolo_label_file = os.path.join(yolo_labels_dir, label_file)

        convert_kitti_to_yolo(
            label_file=os.path.join(kitti_labels_dir, label_file),
            image_file=img_file,
            output_file=yolo_label_file
        )
