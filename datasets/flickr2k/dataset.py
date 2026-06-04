from utils import load_image
from utils.env import resolve_datasets_dir
from utils.hashing import write_timestamp_marker

from datasets.preprocess._base import IMAGE_EXTS
from datasets.preprocess.workers import (
    BlurParams,
    BlurNoisePreprocessor,
    ResizeCropPreprocessor,
    TilePreprocessor,
)
from utils import DEVICE

from pathlib import Path
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
    "tiled_pairs",
]


class Flickr2K(Dataset):
    def __init__(
        self,
        subset: Subsets,
        *,
        resize: int = 1024,
        blur_params: BlurParams = BlurParams(kernel_size=5, kernel_sigma=1.0),
        noise_sigma: float = 15.0,
        tile_size: int = 128,
        cache_batch_size: int = 32,
        root: Optional[Path] = None,
        device: torch.device = DEVICE,
    ):
        self.subset = subset
        self.device = device

        self.resize = resize
        self.blur_params = blur_params
        self.noise_sigma = noise_sigma
        self.tile_size = tile_size
        self.cache_batch_size = cache_batch_size

        ensure_fetched(directory=root)
        self.dataset_root = resolve_datasets_dir(root) / "Flickr2K"

        self._cache_subset(subset)
        self.samples = self._index_samples()
        self.groups = self._build_groups()

    # ------------------------------------------------------------------
    # Parameterised folder names
    # ------------------------------------------------------------------

    @property
    def _clean_dir(self) -> Path:
        return self.dataset_root / f"clean_{self.resize}"

    @property
    def _noisy_dir(self) -> Path:
        k = self.blur_params.kernel_size
        s = self.blur_params.kernel_sigma
        n = self.noise_sigma
        return self.dataset_root / f"noisy_k{k}_s{s}_n{n}"

    @property
    def _tiled_clean_dir(self) -> Path:
        return self.dataset_root / f"tiled_clean_{self.resize}_{self.tile_size}"

    @property
    def _tiled_noisy_dir(self) -> Path:
        k = self.blur_params.kernel_size
        s = self.blur_params.kernel_sigma
        n = self.noise_sigma
        t = self.tile_size
        return self.dataset_root / f"tiled_noisy_k{k}_s{s}_n{n}_{t}"

    def _pair_folders(self, subset: Subsets) -> Tuple[Path, Path]:
        if subset == "pairs":
            return self._clean_dir, self._noisy_dir
        if subset == "tiled_pairs":
            return self._tiled_clean_dir, self._tiled_noisy_dir
        raise ValueError(f"{subset!r} is not a pair subset")

    def _subset_dir(self, subset: Subsets) -> Path:
        return {
            "clean": self._clean_dir,
            "noisy": self._noisy_dir,
            "tiled_clean": self._tiled_clean_dir,
            "tiled_noisy": self._tiled_noisy_dir,
        }[subset]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        returnval = lambda x: load_image(x).cpu().float() / 255.0

        if "pairs" in self.subset:
            noisy_path, clean_path = self.samples[index]
            return (
                returnval(noisy_path),
                returnval(clean_path),
            )

        return returnval(self.samples[index])

    def _index_samples(self) -> List[Path] | List[Tuple[Path, Path]]:
        list_images = lambda d: sorted(
            p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )

        if "pairs" in self.subset:
            clean_dir, noisy_dir = self._pair_folders(self.subset)
            clean_files = list_images(clean_dir)
            noisy_files = list_images(noisy_dir)

            if len(clean_files) != len(noisy_files):
                raise RuntimeError(
                    f"Clean/noisy file count mismatch: {len(clean_files)} vs {len(noisy_files)}"
                )

            return list(zip(noisy_files, clean_files))

        return list_images(self._subset_dir(self.subset))

    def _build_groups(self) -> Optional[List[str]]:
        if not self.subset.startswith("tiled_"):
            return None

        groups = []
        for sample in self.samples:
            path = sample[0] if isinstance(sample, tuple) else sample
            groups.append(path.stem.rsplit("_", 2)[0])

        return groups

    # ------------------------------------------------------------------
    # Cache orchestration
    # ------------------------------------------------------------------

    def _cache_subset(self, subset: Subsets):
        raw = self.dataset_root / "raw"

        clean = self._clean_dir
        noisy = self._noisy_dir
        tiled_clean = self._tiled_clean_dir
        tiled_noisy = self._tiled_noisy_dir

        steps = {
            "clean": [
                lambda: self._cache_clean(raw, clean),
            ],
            "noisy": [
                lambda: self._cache_clean(raw, clean),
                lambda: self._cache_noisy(clean, noisy),
            ],
            "pairs": [
                lambda: self._cache_clean(raw, clean),
                lambda: self._cache_noisy(clean, noisy),
            ],
            "tiled_clean": [
                lambda: self._cache_clean(raw, clean),
                lambda: self._cache_tiled(clean, tiled_clean),
            ],
            "tiled_noisy": [
                lambda: self._cache_clean(raw, clean),
                lambda: self._cache_noisy(clean, noisy),
                lambda: self._cache_tiled(noisy, tiled_noisy),
            ],
            "tiled_pairs": [
                lambda: self._cache_clean(raw, clean),
                lambda: self._cache_noisy(clean, noisy),
                lambda: self._cache_tiled(clean, tiled_clean),
                lambda: self._cache_tiled(noisy, tiled_noisy),
            ],
        }

        if subset not in steps:
            raise ValueError(f"Unknown subset: {subset!r}")

        for step in steps[subset]:
            step()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _is_done(self, path: Path) -> bool:
        return (path / ".done").exists()

    def _mark_done(self, path: Path) -> None:
        write_timestamp_marker(path / ".done")

    def _cache_clean(self, src: Path, dst: Path) -> None:
        if self._is_done(dst):
            return
        ResizeCropPreprocessor(self.resize).run(
            src,
            dst,
            batch_size=1,
            device=self.device,
        )
        assert len(list(src.iterdir())) - 1 == len(list(dst.iterdir()))
        self._mark_done(dst)

    def _cache_noisy(self, src: Path, dst: Path) -> None:
        if self._is_done(dst):
            return
        BlurNoisePreprocessor(self.blur_params, self.noise_sigma).run(
            src,
            dst,
            batch_size=self.cache_batch_size,
            device=self.device,
        )
        assert len(list(src.iterdir())) - 1 == len(list(dst.iterdir()))
        self._mark_done(dst)

    def _cache_tiled(self, src: Path, dst: Path) -> None:
        if self._is_done(dst):
            return
        TilePreprocessor(self.tile_size).run(
            src,
            dst,
            batch_size=self.cache_batch_size,
            device=self.device,
        )
        self._mark_done(dst)

