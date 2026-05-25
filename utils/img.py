from pathlib import Path

import torch
from torchvision.io import read_image, write_png


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    # uint8 tensor, [C, H, W]
    return read_image(str(path)).to(device)


def save_image(img: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(img.cpu(), str(path))