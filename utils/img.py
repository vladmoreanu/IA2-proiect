from pathlib import Path

import torch
from torchvision.io import read_image, write_png


def load_image(path: Path) -> torch.Tensor:
    # uint8 tensor, [C, H, W]
    return read_image(str(path))


def save_image(img: torch.Tensor, path: Path):
    '''
    Do make sure the tensor is on cpu before running this...
    '''
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(img, str(path))