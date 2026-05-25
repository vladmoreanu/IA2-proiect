from utils import load_image
from utils.env import resolve_datasets_dir
from utils.hashing import write_timestamp_marker
from datasets.preprocess import noise, tile

from pathlib import Path
import random
from typing import Optional, Literal, List, Tuple

import torch
from torch.utils.data import Dataset

from .fetch import ensure_fetched


Subsets = Literal[
    "clean",
    "noisy",
    "pairs",
    "tiled_clean",
    "tiled_noisy",
    "tiled_pairs"
]


random.seed(32)
ker_size = random.choice([3, 5, 7, 9])
ker_sigma = random.uniform(0.1, 2.0)
noise_sigma = random.uniform(0, 50)


class Flickr2K(Dataset):
    def __init__(
        self,
        subset: Subsets,
        *,
        root: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        self.subset = subset
        self.device = device or torch.device("cpu")

        ensure_fetched(directory=root)

        self.dataset_root = resolve_datasets_dir(root) / "Flickr2K"

        self.subset_dir = self._ensure_subset(subset)

        self.samples = self._index_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        if "pairs" in self.subset:
            noisy_path, clean_path = self.samples[index]

            noisy = load_image(noisy_path, self.device)
            clean = load_image(clean_path, self.device)

            return noisy, clean

        img = load_image(self.samples[index], self.device)
        return img

    def _ensure_subset(self, subset: Subsets) -> Path:
        def is_done(path) -> bool:
            return (path / ".done").exists()
        
        def mark_done(path: Path) -> None:
            write_timestamp_marker(path / ".done")

        def ensure_noisy(clean_path: Path, noisy_path: Path) -> None:
            if not is_done(noisy_path):
                noise(
                    clean_path,
                    noisy_path,
                    ker_size,
                    ker_sigma,
                    noise_sigma,
                    self.device,
                )
                mark_done(noisy_path)

        def ensure_tiled(src: Path, dst: Path) -> None:
            if not is_done(dst):
                tile(
                    src,
                    dst,
                    128,
                    self.device,
                )
                mark_done(dst)
        clean = self.dataset_root / "clean"
        noisy = self.dataset_root / "noisy"

        tiled_clean = self.dataset_root / "tiled_clean"
        tiled_noisy = self.dataset_root / "tiled_noisy"

        if subset == "clean":
            if not is_done(clean):
                raise FileNotFoundError("Clean dataset missing")
            return clean

        if subset == "noisy":
            ensure_noisy(clean, noisy)
            return noisy

        if subset == "pairs":
            ensure_noisy(clean, noisy)
            return clean, noisy

        if subset == "tiled_clean":
            ensure_tiled(clean, tiled_clean)
            return tiled_clean

        if subset == "tiled_noisy":
            ensure_noisy(clean, noisy)
            ensure_tiled(noisy, tiled_noisy)
            return tiled_noisy

        if subset == "tiled_pairs":
            ensure_noisy(clean, noisy)
            ensure_tiled(clean, tiled_clean)
            ensure_tiled(noisy, tiled_noisy)
            return tiled_clean, tiled_noisy

        raise ValueError(f"Unknown subset: {subset}")
    
    def _index_samples(self) -> List[Path] | List[Tuple[Path, Path]]:
        PAIR_SUBSETS = {
            "pairs": ("clean", "noisy"),
            "tiled_pairs": ("tiled_clean", "tiled_noisy"),
        }

        if self.subset in PAIR_SUBSETS:
            clean_name, noisy_name = PAIR_SUBSETS[self.subset]

            clean_dir = self.dataset_root / clean_name
            noisy_dir = self.dataset_root / noisy_name

            clean_files = sorted(clean_dir.iterdir())
            noisy_files = sorted(noisy_dir.iterdir())

            if len(clean_files) != len(noisy_files):
                raise RuntimeError(
                    f"{self.subset}: clean/noisy file count mismatch "
                    f"({len(clean_files)} != {len(noisy_files)})"
                )

            return list(zip(noisy_files, clean_files))

        subset_dir = self.dataset_root / self.subset
        return sorted(subset_dir.iterdir())