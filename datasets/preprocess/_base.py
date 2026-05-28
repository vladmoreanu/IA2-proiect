from utils import load_image, save_image, DEVICE

from abc import ABC, abstractmethod
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


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
    @abstractmethod
    def transform(self, images: torch.Tensor) -> torch.Tensor:
        ...

    def total_output_files(self, ds: "ImageDirDataset") -> int:
        return len(ds)

    def save_batch(
        self,
        result: torch.Tensor,
        batch_names: list[str],
        output_dir: Path,
        save_pool: ThreadPoolExecutor,
    ) -> set:
        futures = set()
        for img, name in zip(result, batch_names):
            futures.add(save_pool.submit(save_image, img, output_dir / name))
        return futures

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

        main_bar = tqdm(
            total=len(loader),
            desc=f"{self.__class__.__name__}:{input_dir.name}",
            ncols=80,
            dynamic_ncols=True,
            leave=True,
        )

        save_bar = tqdm(
            total=self.total_output_files(ds),
            desc="Saved",
            ncols=80,
            dynamic_ncols=True,
            leave=True,
        )

        save_pool = ThreadPoolExecutor(max_workers=4)
        pending = set()

        try:
            for batch_images, batch_names in loader:
                batch_images = batch_images.to(device, non_blocking=pin)
                with torch.inference_mode():
                    result = self.transform(batch_images)
                assert result.device == batch_images.device, (
                    f"transform() returned a tensor on {result.device}, expected {batch_images.device}. "
                    "Make sure all operations in transform() stay on the input device."
                )
                result = result.detach().cpu()
                pending |= self.save_batch(result, batch_names, output_dir, save_pool)

                while len(pending) >= MAX_PENDING:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in done:
                        f.result()
                        save_bar.update(1)
                    pending = set(pending)

                main_bar.update(1)
            main_bar.close()

        except KeyboardInterrupt:
            print("KeyboardInterrupt received. Cancelling pending saves...")
            for f in pending:
                f.cancel()
            raise

        finally:
            while pending:
                done = {f for f in pending if f.done()}
                for f in done:
                    f.result()
                    save_bar.update(1)
                pending -= done
                save_bar.refresh()
            save_pool.shutdown(wait=True)
            save_bar.close()