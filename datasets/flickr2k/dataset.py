from utils import load_image
from utils.env import resolve_datasets_dir
from utils.hashing import write_timestamp_marker

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

_PAIR_SUBSETS = {
    "pairs":       ("clean", "noisy"),
    "tiled_pairs": ("tiled_clean", "tiled_noisy"),
}


class Flickr2K(Dataset):
    def __init__(
        self,
        subset: Subsets,
        *,
        resize: int = 1024,
        blur_params: BlurParams = BlurParams(kernel_size=15, kernel_sigma=5.0),
        noise_sigma: float = 25.0,
        tile_size: int = 64,
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        # Always load onto CPU — this is called inside DataLoader workers
        # which cannot share CUDA contexts.
        if "pairs" in self.subset:
            noisy_path, clean_path = self.samples[index]
            return (
                load_image(noisy_path).cpu(),
                load_image(clean_path).cpu(),
            )

        return load_image(self.samples[index]).cpu()

    def _index_samples(self) -> List[Path] | List[Tuple[Path, Path]]:
        if self.subset in _PAIR_SUBSETS:
            clean_name, noisy_name = _PAIR_SUBSETS[self.subset]
            clean_files = sorted((self.dataset_root / clean_name).iterdir())
            noisy_files = sorted((self.dataset_root / noisy_name).iterdir())

            if len(clean_files) != len(noisy_files):
                raise RuntimeError(
                    f"Clean/noisy file count mismatch: {len(clean_files)} vs {len(noisy_files)}"
                )

            return list(zip(noisy_files, clean_files))

        return sorted((self.dataset_root / self.subset).iterdir())

    def _build_groups(self) -> Optional[List[str]]:
        if not self.subset.startswith("tiled_"):
            return None

        groups = []
        for sample in self.samples:
            path = sample[0] if isinstance(sample, tuple) else sample
            groups.append(path.stem.rsplit("_", 2)[0])

        return groups

    def _cache_subset(self, subset: Subsets):
        raw        = self.dataset_root / "raw"
        clean      = self.dataset_root / "clean"
        noisy      = self.dataset_root / "noisy"
        tiled_clean = self.dataset_root / "tiled_clean"
        tiled_noisy = self.dataset_root / "tiled_noisy"

        steps = {
            "clean":       [lambda: self._cache_clean(raw, clean)],
            "noisy":       [lambda: self._cache_clean(raw, clean),
                            lambda: self._cache_noisy(clean, noisy)],
            "pairs":       [lambda: self._cache_clean(raw, clean),
                            lambda: self._cache_noisy(clean, noisy)],
            "tiled_clean": [lambda: self._cache_clean(raw, clean),
                            lambda: self._cache_tiled(clean, tiled_clean)],
            "tiled_noisy": [lambda: self._cache_clean(raw, clean),
                            lambda: self._cache_noisy(clean, noisy),
                            lambda: self._cache_tiled(noisy, tiled_noisy)],
            "tiled_pairs": [lambda: self._cache_clean(raw, clean),
                            lambda: self._cache_noisy(clean, noisy),
                            lambda: self._cache_tiled(clean, tiled_clean),
                            lambda: self._cache_tiled(noisy, tiled_noisy)],
        }

        if subset not in steps:
            raise ValueError(f"Unknown subset: {subset!r}")

        for step in steps[subset]:
            step()

    def _is_done(self, path: Path) -> bool:
        return (path / ".done").exists()

    def _mark_done(self, path: Path) -> None:
        write_timestamp_marker(path / ".done")

    def _cache_clean(self, src: Path, dst: Path) -> None:
        if self._is_done(dst):
            return
        ResizeCropPreprocessor(self.resize).run(
            src, dst,
            batch_size=1,
            device=self.device,
        )
        assert len(list(src.iterdir())) == len(list(dst.iterdir()))-1
        self._mark_done(dst)

    def _cache_noisy(self, src: Path, dst: Path) -> None:
        if self._is_done(dst):
            return
        BlurNoisePreprocessor(self.blur_params, self.noise_sigma).run(
            src, dst,
            batch_size=self.cache_batch_size,
            device=self.device,
        )
        assert len(list(src.iterdir())) == len(list(dst.iterdir()))-1
        self._mark_done(dst)

    def _cache_tiled(self, src: Path, dst: Path) -> None:
        if self._is_done(dst):
            return
        TilePreprocessor(self.tile_size).run(
            src, dst,
            batch_size=self.cache_batch_size,
            device=self.device,
        )
        self._mark_done(dst)
