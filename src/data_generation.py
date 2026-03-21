import os
import cv2
import numpy as np
import random

path_normal = r"E:\baze de date\Flickr2K\normal_images"
path_noise = r"E:\baze de date\Flickr2K\noise_images"

# Dataset link: https://www.kaggle.com/datasets/daehoyang/flickr2k


def add_noise(image_path, output_dir, output_name=None, kernel_size=5, kernel_sigma=0.0, noise_sigma=25.0):
    original_img = cv2.imread(image_path)
    img_name = os.path.basename(image_path)

    blurred = cv2.GaussianBlur(original_img, (kernel_size, kernel_size), kernel_sigma)  # blur
    noise = np.random.normal(0, noise_sigma, original_img.shape).astype(np.float32)  # noise
    noisy_blurred = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if output_name:
        cv2.imwrite(os.path.join(output_dir, output_name), noisy_blurred)
    else:
        cv2.imwrite(os.path.join(output_dir, img_name), noisy_blurred)


for image_name in os.listdir(path_normal):
    ker_size = random.choice([3, 5, 7, 9])
    ker_sigma = random.uniform(0.1, 2.0)
    nos_sigma = random.uniform(0, 50)

    add_noise(image_path=os.path.join(path_normal, image_name), output_dir=path_noise,
              kernel_size=ker_size, kernel_sigma=ker_sigma, noise_sigma=nos_sigma)
