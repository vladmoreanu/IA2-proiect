from lighter.callbacks import MonitorCallback

import torch
import os

class Checkpoint(MonitorCallback):
    def __init__(
        self,
        filepath,
        monitor="val_loss",
        # verbose=0,
        save_best_only=False,
        # save_weights_only=False,
        mode="auto",
        # save_freq="epoch",
        initial_value_threshold=None,
    ):
        '''
        Save the model

        `filepath` may contain placeholders such as
        `{epoch:02d}`,`{batch:02d}` and `{loss:.2f}`. A mismatch between
        logged metrics and the path's placeholders can cause formatting to
        fail.
        '''
        super().__init__(monitor, mode, initial_value_threshold)

        self.filepath       = filepath
        self.save_best_only = save_best_only
        self.last_saved     = None

    def on_epoch_begin(self, epoch, logs=None):
        if self.last_saved:
            self._model.load(self.last_saved)

    def on_epoch_end(self, epoch, logs=None):
        if self._should_save(logs):
            self._save_model(epoch=epoch, batch=None, logs=logs)

    def _should_save(self, logs):
        logs = logs or {}
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
        
    def _save_model(self, epoch, batch, logs):
        file_path = self._get_file_path(epoch, batch, logs)
        self._model.save(file_path)
        self.last_saved = file_path
        # output_dir, _ = os.path.split(filepath)
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # torch.save(self._model.state_dict(), filepath)

    def _get_file_path(self, epoch, batch, logs):
        try:
            # `filepath` may contain placeholders such as
            # `{epoch:02d}`,`{batch:02d}` and `{mape:.2f}`. A mismatch between
            # logged metrics and the path's placeholders can cause formatting to
            # fail.
            if batch is None or "batch" in logs:
                file_path = self.filepath.format(epoch=epoch + 1, **logs)
            else:
                file_path = self.filepath.format(
                    epoch=epoch + 1, batch=batch + 1, **logs
                )
        except KeyError as e:
            raise KeyError(
                f'Failed to format this callback filepath: "{self.filepath}". '
                f"Reason: {e}"
            )
        return file_path
