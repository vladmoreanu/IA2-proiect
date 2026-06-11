from .monitorcallback import MonitorCallback

import re
from pathlib import Path
from typing import Optional

import torch

from typing import Union, List

class BackupRestore(MonitorCallback):
    _BATCH_TMPL = "batch-{epoch:04d}-{batch:06d}.pth"
    _BATCH_RE   = re.compile(r"^batch-(\d+)-(\d+)\.pth$")

    def __init__(
        self,
        dirpath: Union[str , Path],
        monitor: str = "val_loss",
        mode: str = "auto",
        min_delta: float = 0,
        baseline=None,
        save_every_n_batches: int = 0,
        max_batch_saves: Optional[int] = 3,
        restore_on_train_begin: bool = True,
    ) -> None:
        super().__init__(monitor=monitor, mode=mode, baseline=baseline, min_delta=min_delta)
        self.dirpath                = Path(dirpath)
        self.save_every_n_batches   = save_every_n_batches
        self.max_batch_saves        = max_batch_saves
        self.restore_on_train_begin = restore_on_train_begin

        self._epoch: int              = 0
        self._batch: int              = 0
        self._batch_files: list[Path] = []

    def _state(self, epoch: int, batch: int) -> dict:
        return {
            "epoch":       epoch,
            "batch":       batch,
            "best_metric": self.best,
            "params":      self.params or {},
            "model":       self._model.state_dict(),
            "optimizer":   self._model.optimizer.state_dict(),
        }

    def _save(self, path: Path, epoch: int, batch: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._state(epoch, batch), path)
        print(f"[BackupRestore] Saved {path} (epoch {epoch}, batch {batch})")

    def _prune(self, files: List[Path], max_saves: Optional[int]) -> List[Path]:
        if max_saves is None:
            return files
        while len(files) > max_saves:
            oldest = files.pop(0)
            if oldest.exists():
                oldest.unlink()
                print(f"[BackupRestore] Removed old checkpoint: {oldest}")
        return files

    def _latest_checkpoint(self) -> Optional[Path]:
        if not self.dirpath.exists():
            return None
        best_epoch, best_batch, best_path = -1, -1, None
        for f in self.dirpath.iterdir():
            m = self._BATCH_RE.match(f.name)
            if m:
                ep, ba = int(m.group(1)), int(m.group(2))
                if (ep, ba) > (best_epoch, best_batch):
                    best_epoch, best_batch, best_path = ep, ba, f
        return best_path

    def restore(self, path: Optional[Union[str, Path]] = None) -> dict:
        if path is None:
            path = self._latest_checkpoint()
        if path is None:
            raise FileNotFoundError(f"No checkpoint found in {self.dirpath}")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=self._model.device)
        print(f"[BackupRestore] Restoring from {path} (epoch {ckpt['epoch']}, batch {ckpt['batch']})")

        self._model.load_state_dict(ckpt["model"])
        self._model.optimizer.load_state_dict(ckpt["optimizer"])
        self._epoch = ckpt["epoch"]
        self.best   = ckpt.get("best_metric", self.best)
        return ckpt

    def on_train_begin(self, logs=None) -> None:
        self.dirpath.mkdir(parents=True, exist_ok=True)
        if self.restore_on_train_begin:
            latest = self._latest_checkpoint()
            if latest is not None:
                self.restore(latest)
            else:
                print(f"[BackupRestore] No existing checkpoint in {self.dirpath}, starting fresh.")

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        self._epoch = epoch
        self._batch = 0

    def on_train_batch_end(self, batch: int, logs=None) -> None:
        self._batch = batch
        if self.save_every_n_batches < 1:
            return
        if (batch + 1) % self.save_every_n_batches != 0:
            return
        path = self.dirpath / self._BATCH_TMPL.format(epoch=self._epoch, batch=batch)
        self._save(path, epoch=self._epoch, batch=batch)
        self._batch_files.append(path)
        self._batch_files = self._prune(self._batch_files, self.max_batch_saves)

    def on_train_end(self, logs=None) -> None:
        print(f"[BackupRestore] Training complete. Best {self.monitor}: {self.best:.6f}. Checkpoints in: {self.dirpath}")