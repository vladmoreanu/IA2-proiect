from lighter.callbacks import Callback

from pathlib import Path
import csv

class CSVLogger(Callback):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        hist = self._model.history.history
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(hist.keys())
            writer.writerows(zip(*hist.values()))
