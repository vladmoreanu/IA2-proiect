import torch
import matplotlib.pyplot as plt

from modeling.metrics import mse, psnr

from utils import DEVICE
from datasets import Flickr2K
from datasets.preprocess.funcs import tile, untile

TILE_SIZE = 128
INDEX = 0

def main():
    dataset = Flickr2K(subset="clean", tile_size=TILE_SIZE, device=DEVICE)
    image = dataset[INDEX].unsqueeze(0)  # (1, C, H, W)

    n, c, h, w = image.shape
    tiles_h = h // TILE_SIZE
    tiles_w = w // TILE_SIZE

    tiled    = tile(image, TILE_SIZE)                       # (1, T, C, tile_size, tile_size)
    restored = untile(tiled, tiles_h, tiles_w)              # (1, C, H, W)

    res = mse(image, restored)
    print(f"MSE:  {res.item():.6f}")

    res = psnr(image, restored)
    print(f"PSNR: {res.item():.3f} dB")

    to_hwc = lambda t: t.squeeze(0).permute(1, 2, 0).cpu().float().clamp(0, 1).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(to_hwc(image))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(to_hwc(restored))
    axes[1].set_title("Tiled -> Untiled")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()