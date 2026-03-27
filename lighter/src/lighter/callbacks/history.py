from lighter.callbacks import Callback

from tqdm import tqdm

class History(Callback):
    def __init__(self):
        super().__init__()
        self.pbar = None

        self.pbar_conf = {
            'ncols'         : 80,
            'unit'          : 'batch',
            'bar_format'    : (
                '{l_bar}{bar}'
                '| {n_fmt}/{total_fmt}'
                ' [{elapsed}, {rate_fmt}{postfix}]'
            ),
        }

    def on_epoch_begin(self, epoch, logs=None):
        pbar_len = len(self.params['train_loader'])
        if (self.params['val_loader'] is not None) \
           & (epoch % self.params['val_freq'] == 0):
            pbar_len += len(self.params['val_loader'])

        self.pbar = tqdm(
            desc=f"Epoch {epoch + 1}/{self.params['epochs']}",
            total=pbar_len,
            **self.pbar_conf,
            )

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.close()
        self._summary(logs)

    def on_train_batch_end(self, batch, logs=None):
        self.pbar.update(1)

    def on_val_batch_end(self, batch, logs=None):
        self.pbar.update(1)

    def on_predict_batch_end(self, batch, logs=None):
        self.pbar.update(1)

    def on_val_begin(self, logs=None):
        pbar_len = len(self.params['data_loader'])

        self.pbar = tqdm(
            total=pbar_len,
            **self.pbar_conf,
            )

    def on_val_end(self, logs=None):
        self.pbar.close()
        self._summary(logs)

    def on_predict_begin(self, logs=None):
        pbar_len = len(self.params['data_loader'])

        self.pbar = tqdm(
            total=pbar_len,
            **self.pbar_conf,
            )

    def on_predict_end(self, logs=None):
        self.pbar.close()

    def _summary(self, logs):
        out_str = '  Summary:'
        for k, v in logs.items():
            v = f'{v:.3f}' if v > 0.01 else f'{v:.2e}'
            out_str += ' {:s}={:s}'.format(k, v)
        print(out_str)
