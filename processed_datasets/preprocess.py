import cv2
import numpy as np
import os

input_dir = "./datasets/images/train"
output_dir = "./processed_datasets/images/train"
os.makedirs(output_dir, exist_ok=True)

sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        # gray scale 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # remove noise
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)

        # sharpen
        sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)

        # save sharpened image
        out_path = os.path.join(output_dir, filename)
        # convert back to BGR for saving
        sharpened_color = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(out_path, sharpened_color)


        print(f'Processed: {filename}')
