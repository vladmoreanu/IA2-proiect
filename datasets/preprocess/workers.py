from ._base import ImageDirDataset, ImagePreprocessor, ThreadPoolExecutor
from utils import save_image, DEVICE
from .funcs import blur, noise, resize_crop, tile

from typing import NamedTuple, List
from pathlib import Path

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader


class BlurParams(NamedTuple):
    kernel_size: int
    kernel_sigma: float


class BlurNoisePreprocessor(ImagePreprocessor):
    def __init__(
        self,
        blur_params: BlurParams,
        noise_sigma: float,
    ):
        self.blur_params = blur_params
        self.noise_sigma = noise_sigma

    def transform(self, images: torch.Tensor) -> torch.Tensor:
        return noise(blur(images, self.blur_params.kernel_size, self.blur_params.kernel_sigma), self.noise_sigma)


class ResizeCropPreprocessor(ImagePreprocessor):
    def __init__(self, target_size: int):
        self.target_size = target_size

    def transform(self, images: torch.Tensor) -> torch.Tensor:
        return resize_crop(images, self.target_size)


class TilePreprocessor(ImagePreprocessor):
    def __init__(self, tile_size: int):
        self.tile_size = tile_size

    def transform(self, images: torch.Tensor) -> torch.Tensor:
        # Returns (N, T, C, tile_size, tile_size) — save_batch handles the
        # extra tile dimension; the base run() loop is otherwise unmodified.
        return tile(images, self.tile_size)

    def total_output_files(self, ds: ImageDirDataset) -> int:
        _, h, w = ds[0][0].shape
        T = (h // self.tile_size) * (w // self.tile_size)
        return len(ds) * T

    def save_batch(
        self,
        result: torch.Tensor,
        batch_names: List[str],
        output_dir: Path,
        save_pool: ThreadPoolExecutor,
    ) -> set:
        # result: (N, T, C, tile_size, tile_size)
        futures = set()
        for img_tiles, name in zip(result, batch_names):
            stem = Path(name).stem
            for tile_idx, tile_img in enumerate(img_tiles):
                futures.add(save_pool.submit(save_image, tile_img, output_dir / f"{stem}_{tile_idx}.png"))
        return futures