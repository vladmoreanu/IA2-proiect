import os
import cv2
from tqdm import tqdm

datasets = os.environ.get("DATASETS")

path_clean = datasets / "Flickr2K/normal_images"
path_noisy = datasets / "Flickr2K/noise_images"

path_clean_output = datasets / "Flickr2K/normal_images_tiles"
path_noisy_output = datasets / "Flickr2K/noise_images_tiles"


def tile_dataset(clean_dir, output_dir, tile_size=128):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filenames = [f for f in os.listdir(clean_dir)]

    for fname in tqdm(filenames):
        img = cv2.imread(os.path.join(clean_dir, fname))

        h, w, _ = img.shape

        for y in range(0, h - tile_size + 1, tile_size):
            for x in range(0, w - tile_size + 1, tile_size):
                tile = img[y: y + tile_size, x: x + tile_size]

                save_name = f"{os.path.splitext(fname)[0]}_{y}_{x}.png"
                cv2.imwrite(os.path.join(output_dir, save_name), tile)


tile_dataset(path_clean, path_clean_output)
tile_dataset(path_noisy, path_noisy_output)

