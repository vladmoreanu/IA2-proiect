from lighter.callbacks import Callback
import matplotlib.pyplot as plt
import os

from tqdm import tqdm

class History(Callback):
    def __init__(self, save_path=None):
        super().__init__()
        self.pbar = None

        self.history_dict = {}
        self.save_path = save_path

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

        if logs:
            for k, v in logs.items():
                if k not in self.history_dict:
                    self.history_dict[k] = []
                self.history_dict[k].append(v)

    def on_train_batch_end(self, batch, logs=None):
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        if not self.history_dict:
            return

        metrics_names = list(set([k.replace('train_', '').replace('val_', '') for k in self.history_dict.keys()]))
        num_metrics = len(metrics_names)

        if num_metrics == 0:
            return

        fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))

        if num_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics_names):
            ax = axes[idx]
            train_data = self.history_dict.get(f'train_{metric}', [])
            val_data = self.history_dict.get(f'val_{metric}', [])

            epochs_length = len(train_data) if train_data else len(val_data)
            epochs_range = range(1, epochs_length + 1)

            if train_data: ax.plot(epochs_range, train_data, label=f'Train {metric}')
            if val_data:   ax.plot(epochs_range, val_data, label=f'Val {metric}')

            ax.set_title(f'Evolution of {metric.upper()}')
            ax.set_xlabel('Epochs')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True)

        plt.tight_layout()

        if self.save_path:
            directory = os.path.dirname(self.save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            plt.savefig(self.save_path)
            print(f"Plot saved to: {self.save_path}")

        plt.show()

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
            v = f'{v:.3f}' if v > 0.01 else f'{v:.3e}'
            out_str += ' {:s}={:s}'.format(k, v)
        print(out_str)
