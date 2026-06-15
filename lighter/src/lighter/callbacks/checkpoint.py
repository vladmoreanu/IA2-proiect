from lighter.callbacks import MonitorCallback

import re
from pathlib import Path

import torch

class Checkpoint(MonitorCallback):
    def __init__(
        self,
        filepath,
        monitor="val_loss",
        # verbose=0,
        save_best_only=False,
        # save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        initial_value_threshold=None,
        restore_on_train_begin=False,
    ):
        """
        Save the model

        `filepath` may contain placeholders such as
        `{epoch:02d}`,`{batch:02d}` and `{val_loss:.2f}`. A mismatch between
        logged metrics and the path's placeholders can cause formatting to
        fail.
        """
        super().__init__(monitor, mode, initial_value_threshold)

        self.filepath = str(filepath)
        self.save_best_only = save_best_only
        self.save_freq = save_freq

        self._epoch = 0
        self._batch = 0

        self.restore_on_train_begin = restore_on_train_begin

        if save_freq != "epoch" and save_freq < 1:
            raise ValueError("save_freq must be either 'epoch' or >= 1.")

    def on_train_begin(self, logs=None):
        if self.restore_on_train_begin:
            path = self._latest_checkpoint()
            if path is not None:
                self.restore(path, monitor_only=True)

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch
        self._batch = 0

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self._batch = batch
        if self.save_freq == "epoch":
            return
        if (batch + 1) % self.save_freq != 0:
            return
        if self._should_save(logs):
            self._save(epoch=self._epoch, batch=batch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.save_freq == "epoch" and self._should_save(logs):
            self._save(epoch=epoch, batch=self._batch, logs=logs)

    def _state(self) -> dict:
        return {
            "epoch": self._epoch,
            "batch": self._batch,
            "best_metric": self.best,
            "params": self.params or {},
            "history" : self._model.history.history if hasattr(self._model, "history") else {},
            "model": self._model.state_dict(),
            "optimizer": self._model.optimizer.state_dict(),
        }

    def _should_save(self, logs):
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                # Something is absolutely fishy if this is the case
                return True
            if self._is_improvement(current, self.best):
                self.best = current
                return True
            else:
                return False
        else:
            return True

    def _save(self, epoch, batch, logs):
        file_path = self._get_file_path(epoch, batch, logs)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._state(), file_path)

    def _get_file_path(self, epoch, batch, logs):
        try:
            if batch is None or "batch" in logs:
                file_path = self.filepath.format(epoch=epoch, **logs)
            else:
                file_path = self.filepath.format(
                    epoch=epoch, batch=batch + 1, **logs
                )
        except KeyError as e:
            raise KeyError(
                f'Failed to format this callback filepath: "{self.filepath}". '
                f"Reason: {e}"
            )
        return Path(file_path)

    def _latest_checkpoint(self):
        filepath = Path(self.filepath)
        dirpath = filepath.parent
        if not dirpath.exists():
            return None

        pattern = re.compile(f"^{re.sub(r'{[^}]*}', '.*', filepath.name)}$")

        latest_mtime = 0
        latest_path = None
        largest_path = None
        n_latest = 0

        for f in dirpath.iterdir():
            if not pattern.match(f.name):
                continue
            mtime = f.stat().st_mtime
            if largest_path is None or str(f) > str(largest_path):
                largest_path = f
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = f
                n_latest = 1
            elif mtime == latest_mtime:
                n_latest += 1

        return latest_path if n_latest == 1 else largest_path

    def restore(self, path, monitor_only = True) -> dict:
        ckpt = torch.load(path, map_location=self._model.device)

        if not monitor_only:
            self._model.load_state_dict(ckpt["model"])
            self._model.optimizer.load_state_dict(ckpt["optimizer"])

        self._epoch = ckpt["epoch"]
        self._batch = ckpt["batch"]
        self.best = ckpt.get("best_metric", self.best)

        return ckpt
