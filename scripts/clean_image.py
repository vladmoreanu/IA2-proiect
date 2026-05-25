import torch
from PIL import Image
import cv2
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from models import DnCNN

path_to_model = r"C:\Users\MSI\PycharmProjects\IA2-proiect\proiect\src\checkpoints\DnCNN_0.pt" # TODO: FIX THIS PATH
path_noisy = datasets / "Flickr2K/exemple/noisy/0002.png"
path_cleaned = datasets / "Flickr2K/exemple/processed/0002_90.png"


def clean_image(model_path, image_path, output_path, tile_size=128, stride=96):
    model = DnCNN()

    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)
    model.eval()

    img = Image.open(image_path).convert('RGB')
    img_tensor = transforms.ToTensor()(img)
    channels, height, width = img_tensor.shape

    canvas = torch.zeros_like(img_tensor)
    weight_mask = torch.zeros((1, height, width))

    window = torch.ones((1, tile_size, tile_size))

    with torch.no_grad():
        for y in tqdm(range(0, height - tile_size + 1, stride)):
            for x in range(0, width - tile_size + 1, stride):
                tile = img_tensor[:, y: y + tile_size, x: x + tile_size].unsqueeze(0)

                output = model(tile).squeeze(0)

                canvas[:, y: y + tile_size, x: x + tile_size] += output * window
                weight_mask[:, y: y + tile_size, x: x + tile_size] += window

    clean_img = canvas / weight_mask
    clean_img = torch.nan_to_num(clean_img, nan=0.0)
    clean_img = clean_img.clamp(0, 1).permute(1, 2, 0).numpy()
    clean_img = (clean_img * 255).astype(np.uint8)
    cv2.imwrite(output_path, cv2.cvtColor(clean_img, cv2.COLOR_RGB2BGR))


clean_image(path_to_model, path_noisy, path_cleaned, stride=90)