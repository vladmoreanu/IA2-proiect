from utils import save_image, DEVICE
from utils.env import resolve_datasets_dir
from modeling import DnCNN
from modeling.metrics import mse, psnr
from datasets import Flickr2K
# from datasets.preprocess.funcs import tile, untile

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


CHECKPOINT = Path("_fit/proiect/DnCNN_0/model_2026-05-28_f0.pth")
OUTPUT_DIR = resolve_datasets_dir() / "Flickr2K/exemple/processed"
TILE_SIZE = 128
STRIDE = 64
BATCH_SIZE = 4
INDEX = 1000


def tile(images: torch.Tensor, tile_size: int, stride: int) -> torch.Tensor:
    # images: (N, C, H, W) float32
    # Returns (N, T, C, tile_size, tile_size)
    n, c, h, w = images.shape

    y_starts = range(0, h - tile_size + 1, stride)
    x_starts = range(0, w - tile_size + 1, stride)

    tiles = [
        images[:, :, y : y + tile_size, x : x + tile_size]
        for y in y_starts
        for x in x_starts
    ]

    # each entry is (N, C, tile_size, tile_size) → stack to (T, N, C, tile_size, tile_size)
    # then permute to (N, T, C, tile_size, tile_size)
    return torch.stack(tiles).permute(1, 0, 2, 3, 4)


def untile(
    tiles: torch.Tensor,
    image_shape: tuple[int, int, int, int],
    tile_size: int,
    stride: int,
) -> torch.Tensor:
    # tiles: (N, T, C, tile_size, tile_size) float32
    # image_shape: (N, C, H, W)
    n, c, h, w = image_shape

    y_starts = range(0, h - tile_size + 1, stride)
    x_starts = range(0, w - tile_size + 1, stride)

    canvas = torch.zeros(n, c, h, w, dtype=tiles.dtype)
    weights = torch.zeros(n, 1, h, w, dtype=tiles.dtype)

    for idx, (y, x) in enumerate((y, x) for y in y_starts for x in x_starts):
        canvas[:, :, y : y + tile_size, x : x + tile_size] += tiles[:, idx]
        weights[:, :, y : y + tile_size, x : x + tile_size] += 1.0

    covered = weights > 0
    canvas[covered.expand_as(canvas)] /= weights.expand_as(canvas)[
        covered.expand_as(canvas)
    ]

    return canvas


class TileDataset(Dataset):
    def __init__(self, tiles: torch.Tensor):
        self.tiles = tiles  # (T, C, tile_size, tile_size)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        return self.tiles[idx]


def main():
    dataset = Flickr2K(subset="pairs", tile_size=TILE_SIZE, device=DEVICE)
    noisy, clean = dataset[INDEX]
    # Both (C, H, W) float32 in [0, 1]; add batch dim for the strided functions.
    noisy = noisy.unsqueeze(0)  # (1, C, H, W)
    clean = clean.unsqueeze(0)

    tiles = tile(noisy, TILE_SIZE, STRIDE)
    # (1, T, C, tile_size, tile_size) → flatten batch+tile for the DataLoader
    n, t, c, th, tw = tiles.shape
    flat_tiles = tiles.view(n * t, c, th, tw)

    loader = DataLoader(
        TileDataset(flat_tiles),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = DnCNN()
    model.device = DEVICE
    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    )
    # model.load(CHECKPOINT)

    predicted_flat = model.predict(loader)  # (T, C, tile_size, tile_size)
    predicted_tiles = predicted_flat.view(
        n, t, c, th, tw
    )  # (1, T, C, tile_size, tile_size)

    # predicted = untile(predicted_tiles, 8, 8)  # (1, C, H, W)
    predicted = untile(predicted_tiles, noisy.shape, TILE_SIZE, STRIDE)

    predicted = predicted.cpu()
    clean = clean.cpu()
    noisy = noisy.cpu()

    print(f"MSE = {mse(clean, predicted).item():.4f}")
    print(f"PSNR = {psnr(clean, predicted).item():.4f} dB")

    to_24bit = lambda x :(x * 255.0).clamp(0, 255).to(torch.uint8).cpu()
    to_hwc = lambda t: t.squeeze(0).permute(1, 2, 0).numpy()

    predicted = to_24bit(predicted)
    clean = to_24bit(clean)
    noisy = to_24bit(noisy)

    fig, axes = plt.subplots(1, 3, figsize=(16, 9))
    axes[0].imshow(to_hwc(clean))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(to_hwc(noisy))
    axes[1].set_title("Noisy input")
    axes[1].axis("off")
    axes[2].imshow(to_hwc(predicted))
    axes[2].set_title("Predicted output")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()
