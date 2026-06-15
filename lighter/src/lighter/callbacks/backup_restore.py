from .checkpoint import Checkpoint

import re
from pathlib import Path
from typing import Optional

import torch

from typing import Union, List

class BackupRestore(Checkpoint):
    _BATCH_TMPL = "batch-{epoch:04d}-{batch:06d}.pth"
    _BATCH_RE   = re.compile(r"^batch-(\d+)-(\d+)\.pth$")

    def __init__(
        self,
        dirpath,
        monitor = "val_loss",
        mode = "auto",
        initial_value_threshold = None,
        save_freq = "epoch",
        max_batch_saves = 2,
        restore_on_train_begin = True,
    ):
        self.dirpath = Path(dirpath)
        filepath = self.dirpath / self._BATCH_TMPL
        super().__init__(
            filepath=filepath,
            monitor=monitor,
            save_best_only=False,
            mode=mode,
            save_freq=save_freq,
            initial_value_threshold=initial_value_threshold,
            restore_on_train_begin=restore_on_train_begin,
        )
        self.save_freq       = save_freq
        self.max_batch_saves = max_batch_saves

        self._file_list = []

    def on_train_begin(self, logs=None):
        if self.restore_on_train_begin:
            latest = self._latest_checkpoint()
            if latest is not None:
                self.restore(latest)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self._batch = batch
        if self.save_freq == "epoch":
            return
        if (batch + 1) % self.save_freq != 0:
            return
        if self._should_save(logs):
            self._save(self._epoch, batch, logs)
            path = self._get_file_path(self._epoch, batch, logs)
            self._file_list.append(path)
            self._prune_files()

    def _prune_files(self):
        while len(self._file_list) > self.max_batch_saves:
            oldest = self._file_list.pop(0)
            if oldest.exists():
                oldest.unlink()

    def _latest_checkpoint(self):
        if not self.dirpath.exists():
            return None
        best_epoch, best_batch, best_path = -1, -1, None
        candidates = []
        for f in self.dirpath.iterdir():
            m = self._BATCH_RE.match(f.name)
            if m:
                ep, ba = int(m.group(1)), int(m.group(2))
                candidates.append((ep, ba, f))
                if (ep, ba) > (best_epoch, best_batch):
                    best_epoch, best_batch, best_path = ep, ba, f

        candidates.sort()
        self._file_list = [f for _, _, f in candidates]
        self._prune_files()  # immediately evict anything beyond max_batch_saves
        return best_path

    def _validate_path(self, path=None):
        if path is None:
            path = self._latest_checkpoint()
        if path is None:
            raise FileNotFoundError(f"No checkpoint found in {self.dirpath}")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        return path

    def peek(self, path=None):
        try:
            path = self._validate_path(path)
        except FileNotFoundError:
            return None
        return torch.load(path, map_location="cpu")

    def restore(self, path=None, monitor_only=False):
        path = self._validate_path(path)
        return super().restore(path, monitor_only=monitor_only)
