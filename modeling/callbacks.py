from lighter.callbacks import Callback

import json
from pathlib import Path


class Reporter(Callback):
    def __init__(self, filepath: Path, train_params):
        super().__init__()
        self.filepath = filepath
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        if not filepath.exists():
            self.report = {
                "train_params": [train_params],
                "results": [],
            }
        else:
            with open(self.filepath, "r", encoding="utf-8") as fp:
                self.report = json.load(fp)
                self.report["train_params"].append(train_params)

    def on_val_end(self, logs=None):
        self.report["results"].append(logs)
        with open(self.filepath, "w", encoding="utf-8") as fp:
            json.dump(self.report, fp, indent=2)
