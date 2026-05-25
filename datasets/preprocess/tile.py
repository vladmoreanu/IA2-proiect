import torch
from pathlib import Path

from tqdm import tqdm

from utils import load_image, save_image



IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def tile(
    input_dir: Path,
    output_dir: Path,
    tile_size: int,
    device: torch.device,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    for img_path in tqdm(images, desc=f"Tiling {input_dir.name}"):
        img = load_image(img_path, device)
        _, h, w = img.shape

        for y in range(0, h - tile_size + 1, tile_size):
            for x in range(0, w - tile_size + 1, tile_size):
                tile = img[:, y:y + tile_size, x:x + tile_size]
                name = f"{img_path.stem}_{y}_{x}.png"
                save_image(tile, output_dir / name)
