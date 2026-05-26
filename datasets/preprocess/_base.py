from utils import load_image, save_image, DEVICE

from abc import ABC, abstractmethod
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


MAX_PENDING = 8


class ImageDirDataset(Dataset):
    def __init__(self, input_dir: Path):
        self.paths = [
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        # Workers cannot share CUDA contexts — always load onto CPU here.
        # The main loop moves the batch to the target device after collation.
        return load_image(self.paths[idx]).cpu(), self.paths[idx].name


class ImagePreprocessor(ABC):
    """
    Subclass this and implement `transform`. The DataLoader boilerplate
    is handled here — adding a new preprocessing mode means writing
    only the transform logic.

    Usage:
        BlurNoisePreprocessor(blur_params, noise_sigma).run(input_dir, output_dir)
    """

    @abstractmethod
    def transform(self, images: torch.Tensor) -> torch.Tensor:
        ...

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        *,
        batch_size: int = 32,
        num_workers: int = 4,
        prefetch_factor: int = 4,
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

        save_bar = tqdm(
            total=len(ds),
            desc='Saved',
            ncols=80,
            dynamic_ncols=True,
            position=1,
            leave=True,
        )

        main_bar = tqdm(
            loader,
            desc=f"{self.__class__.__name__} — {input_dir.name}",
            ncols=80,
            dynamic_ncols=True,
            position=0,
        )

        save_pool = ThreadPoolExecutor(max_workers=4)
        pending = set()

        try:
            for batch_images, batch_names in main_bar:
                batch_images = batch_images.to(device, non_blocking=pin)
                result = self.transform(batch_images)
                assert result.device == batch_images.device, (
                    f"transform() returned a tensor on {result.device}, expected {batch_images.device}. "
                    "Make sure all operations in transform() stay on the input device."
                )
                with torch.inference_mode():
                    result = self.transform(batch_images)
                for img, name in zip(result, batch_names):
                    future = save_pool.submit(save_image, img, output_dir / name)
                    pending.add(future)
                
                # Drain completed futures without blocking
                done = {f for f in pending if f.done()}
                for f in done:
                    f.result() # surface exceptions
                    save_bar.update(1)
                pending -= done

        except KeyboardInterrupt:
            print("KeyboardInterrupt received. Cancelling pending saves...")
            for f in pending:
                f.cancel() # cancels only if not started
            raise

        finally:
            for f in pending:
                f.result()
                save_bar.update(1)
            save_pool.shutdown(wait=True)
            save_bar.close()
