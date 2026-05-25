from utils import load_image, save_image

from pathlib import Path
import random

from tqdm import tqdm
import torch
from torchvision.transforms.functional import gaussian_blur


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def noise(
    input_dir: Path,
    output_dir: Path,
    kernel_size: int,
    kernel_sigma: float,
    noise_sigma: float,
    device: torch.device,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    for img_path in tqdm(images, desc=f"Noising {input_dir.name}"):
        img = load_image(img_path, device)

        blurred = gaussian_blur(
            img,
            kernel_size=[kernel_size, kernel_size],
            sigma=kernel_sigma,
        )

        noise = torch.randn_like(blurred, dtype=torch.float32) * noise_sigma

        noisy = torch.clamp(
            blurred.to(torch.float32) + noise,
            0,
            255,
        ).to(torch.uint8)

        save_image(noisy, output_dir / img_path.name)

