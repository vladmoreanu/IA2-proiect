from lighter.callbacks import Callback

import torch

class Checkpoint(Callback):
    def __init__(
        self,
        filepath,
        # monitor="val_loss",
        # verbose=0,
        # save_best_only=False,
        # save_weights_only=False,
        # mode="auto",
        # save_freq="epoch",
        # initial_value_threshold=None,
    ):
        super().__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        self._save_model()

    def _save_model(self):
        torch.save(self._model.state_dict(), self.filepath)