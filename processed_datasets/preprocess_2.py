import cv2
import os
from pathlib import Path

# input_dir = "/Users/rin/Documents/畢業專題/YOLO/yolo-3/datasets/images/train"
# output_dir = "/Users/rin/Documents/畢業專題/YOLO/yolo-3/datasets/preprocessed_images/train"

input_dir = "/Users/rin/Documents/畢業專題/YOLO/yolo-3/datasets/images/val"
output_dir = "/Users/rin/Documents/畢業專題/YOLO/yolo-3/preprocessed_datasets_2/images/val"
Path(output_dir).mkdir(parents=True, exist_ok=True)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    cv2.imwrite(os.path.join(output_dir, img_name), enhanced)
    