from ._base import (
    DEVICE,
    ImageDirDataset,
    ImagePreprocessor,
    ThreadPoolExecutor,
    wait,
    FIRST_COMPLETED,
    MAX_PENDING,
)
from utils import save_image
from .funcs import *

from typing import NamedTuple
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
        blur_params: BlurParams = BlurParams(kernel_size=15, kernel_sigma=5.0),
        noise_sigma: float = 25.0,
    ):
        self.blur_params = blur_params
        self.noise_sigma = noise_sigma

    def transform(self, images: torch.Tensor) -> torch.Tensor:
        return noise(blur(images, self.blur_params.kernel_size, self.blur_params.kernel_sigma), self.noise_sigma)


class ResizeCropPreprocessor(ImagePreprocessor):
    def __init__(self, target_size: int = 1024):
        self.target_size = target_size

    def transform(self, images: torch.Tensor) -> torch.Tensor:
        return resize_crop(images, self.target_size)


class TilePreprocessor:
    """
    Tiles images into (tile_size x tile_size) patches and saves each patch
    as a separate file named {stem}_{top}_{left}.png.

    Does not subclass ImagePreprocessor because the 1-to-many output pattern
    (one image -> T tiles) doesn't fit the base class's 1-to-1 save loop.
    """

    def __init__(self, tile_size: int = 64):
        self.tile_size = tile_size

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        *,
        batch_size: int = 32,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        device: torch.device = DEVICE,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        ds = ImageDirDataset(input_dir)
        pin = device.type == "cuda"
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=pin,
            persistent_workers=num_workers > 0,
        )

        main_bar = tqdm(
            loader,
            desc=f"{self.__class__.__name__} — {input_dir.name}",
            ncols=80,
            dynamic_ncols=True,
            position=0,
        )

        save_pool = ThreadPoolExecutor(max_workers=num_workers)
        pending = set()

        try:
            for batch_images, batch_names in main_bar:
                batch_images = batch_images.to(device, non_blocking=pin)
                result = tile(batch_images, self.tile_size)
                assert result.device == batch_images.device, (
                    f"transform() returned a tensor on {result.device}, expected {batch_images.device}. "
                    "Make sure all operations in transform() stay on the input device."
                )
                # result: (N, T, C, tile_size, tile_size)
                result = result.detach().cpu()
                for img_tiles, name in zip(result, batch_names):
                    stem = Path(name).stem
                    for tile_idx, tile_img in enumerate(img_tiles):
                        future = save_pool.submit(save_image, tile_img, output_dir / f"{stem}_{tile_idx}.png")
                        pending.add(future)
                # Drain completed futures without blocking
                done = {f for f in pending if f.done()}
                for f in done:
                    f.result() # surface exceptions
                pending -= done

        except KeyboardInterrupt:
            print("KeyboardInterrupt received. Cancelling pending saves...")
            for f in pending:
                f.cancel()   # cancels only if not started
            raise
        finally:
            for f in pending:
                f.result()
            save_pool.shutdown(wait=True)
