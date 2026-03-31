from lighter.callbacks import Callback

import os
import pandas as pd

class CSVLogger(Callback):
    def __init__(self, path):
        super().__init__()
        self.path = path
    
    def on_epoch_end(self, epoch, logs=None):
        log = {}
        for k, v in logs.items():
            log[k] = [v]
        df = pd.DataFrame(log)
        df.to_csv(
            self.path,
            mode='a',
            header=not os.path.exists(self.path),
            index=False,
        )