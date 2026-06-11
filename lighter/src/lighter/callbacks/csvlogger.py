from lighter.callbacks import Callback

import os
import csv

class CSVLogger(Callback):
    def __init__(self, path):
        super().__init__()
        self.path = path
        dirs, _ = os.path.split(self.path)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        if os.path.exists(self.path):
            os.remove(self.path)

    def on_epoch_end(self, epoch, logs=None):
        write_header = not os.path.exists(self.path)
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=logs.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(logs)
