from .callback import Callback

from typing import List

class CallbackList(Callback):
    def __init__(
        self,
        callbacks : List[Callback] = None,
        model = None,
        **params,
    ):
        self.callbacks = callbacks

        if not model:
            raise ValueError('A model must be provided')
        super().set_model(model)
        for callback in self.callbacks:
            callback.set_model(model)

        if params:
            super().set_params(params)
            for callback in self.callbacks:
                callback.set_params(params)

    def on_epoch_begin(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_train_batch_end(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_train_batch_end(batch, logs=logs)

    def on_val_batch_end(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_val_batch_end(batch, logs=logs)

    def on_predict_batch_end(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_predict_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_val_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_val_begin(logs)

    def on_val_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_val_end(logs)

    def on_predict_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_predict_begin(logs)

    def on_predict_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_predict_end(logs)
