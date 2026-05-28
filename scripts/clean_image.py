import torch
from torch.utils.data import Dataset, DataLoader

from pathlib import Path

from utils import save_image, DEVICE
from utils.env import resolve_datasets_dir
from models import DnCNN
from datasets import Flickr2K

from lighter.metrics import PSNR
from torch.nn.functional import mse_loss

CHECKPOINT = Path(
    "/home/vladm/Facultate/S2/IA2-proiect/_fit/proiect/DnCNN_0/model_2026-05-28_f0.pt"
)
OUTPUT_DIR = resolve_datasets_dir() / "Flickr2K/exemple/processed"
TILE_SIZE = 128
STRIDE = 96
BATCH_SIZE = 4
INDEX = 200


class TileDataset(Dataset):
    def __init__(self, tiles: torch.Tensor):
        self.tiles = tiles  # (T, C, tile_size, tile_size)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        return self.tiles[idx]


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

    predicted = untile(predicted_tiles, noisy.shape, TILE_SIZE, STRIDE)  # (1, C, H, W)

    print(mse_loss(clean, predicted))
    psnr = PSNR()
    psnr.update(clean, predicted)
    print(psnr.result())

    predicted = (predicted * 255.0).to(torch.uint8)
    clean = (clean * 255.0).to(torch.uint8)
    noisy = (noisy * 255.0).to(torch.uint8)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_image(predicted.squeeze(0), OUTPUT_DIR / "predicted.png")
    save_image(clean.squeeze(0), OUTPUT_DIR / "clean.png")
    save_image(noisy.squeeze(0), OUTPUT_DIR / "noisy.png")


if __name__ == "__main__":
    main()
