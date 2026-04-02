from .metrics import Metric, Loss
from .callbacks import CallbackList

import torch

import os

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def save(
        self,
        filepath,
    ):
        output_dir, _ = os.path.split(filepath)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.state_dict(), filepath)

    def load(
        self,
        filepath,
    ):
        self.load_state_dict(torch.load(
            filepath,
            map_location=self.device,
            weights_only=True
        ))

    def forward(
        self,
        x
    ):
        return x

    def compile(
        self,
        optimizer=None,
        loss=None,
        # loss_weights=None,
        metrics : list[Metric] = None,
        # weighted_metrics=None,
        # run_eagerly=False,
        # steps_per_execution=1,
        # jit_compile="auto",
        # auto_scale_loss=True,
        device='cpu',
    ):
        self.optimizer  = optimizer
        self.loss_fn    = loss
        self.metrics    = [Loss()] + metrics
        self.device     = device
        
        self.to(self.device)

    def step(
        self,
        data_loader,
        training : bool = False,
    ):
        """
        Single-iteration function, either for training or for testing.

        :param dataloader: torch.utils.data.DataLoader object to iterate over
        """
        self.train() if training else self.eval()
        pfx = 'train_' if training else 'val_'

        for metric in self.metrics:
            metric.reset()

        for idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            if training:
                self.callbacks.on_train_batch_begin(idx)
                
                outputs = self(inputs)
                loss = self.loss_fn(outputs, targets) 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            else:
                self.callbacks.on_val_batch_begin(idx)

                with torch.no_grad():
                    outputs = self(inputs)
                    loss = self.loss_fn(outputs, targets)

            # Metrics update state
            for metric in self.metrics:
                if metric.name == 'loss':
                    metric.update(loss.item())
                else:
                    metric.update(targets, outputs)

            # Create the log (metrics) of this batch
            batch_log = { (pfx + x.name):x.result() for x in self.metrics }
            if training:
                self.callbacks.on_train_batch_end(idx, batch_log)
            else:
                self.callbacks.on_val_batch_end(idx, batch_log)

        # Create and return the log (metrics) of this epoch
        return { (pfx + x.name):x.result() for x in self.metrics }

    def fit(
        self,
        train_loader=None,
        epochs=1,
        # verbose="auto",
        # batch_size=None,
        callbacks=None,
        validation_loader=None,
        # shuffle=True,
        # class_weight=None,
        # sample_weight=None,
        # initial_epoch=1,
        # steps_per_epoch=None,
        # validation_rate=None,
        # validation_batch_size=None,
        validation_freq=1,
    ):
        """
        Train loop.
        """
        self.to(self.device)

        self.callbacks = CallbackList(
            callbacks,
            self,
            train_loader    = train_loader,
            epochs          = epochs,
            val_loader      = validation_loader,
            val_freq        = validation_freq,
        )

        self.callbacks.on_train_begin()

        train_losses = []
        val_losses = []
        
        for e in range(epochs):
            self.callbacks.on_epoch_begin(e)
            
            train_log = self.step(train_loader, training = True)
            if (validation_loader is not None) & (e % validation_freq == 0):
                val_log = self.step(validation_loader)

            train_losses.append(train_log['train_loss'])
            val_losses.append(val_log['val_loss'])

            self.callbacks.on_epoch_end(e, train_log | val_log)

        self.callbacks.on_train_end(train_log | val_log)

        if validation_loader:
            return train_losses, val_losses
        else:
            return train_losses
        
    def evaluate(
        self,
        data_loader=None,
        # batch_size=None,
        # verbose="auto",
        # sample_weight=None,
        # steps=None,
        callbacks=None,
        # return_dict=False,
    ):
        self.to(self.device)

        self.callbacks = CallbackList(
            callbacks,
            self,
            data_loader = data_loader,
        )

        self.callbacks.on_val_begin()

        val_losses = []

        val_log = self.step(data_loader)
        val_losses.append(val_log['val_loss'])

        self.callbacks.on_val_end(val_log)

        return val_losses

    def predict(
        self,
        data_loader,
        # batch_size=None,
        # verbose="auto",
        # steps=None,
        callbacks=None
    ):
        self.to(self.device)

        self.callbacks = CallbackList(
            callbacks,
            self,
            data_loader = data_loader,
        )

        self.eval()

        self.callbacks.on_predict_begin()

        for idx, inputs in enumerate(data_loader):
            inputs = inputs.to(self.device)

            self.callbacks.on_predict_batch_begin(idx)

            with torch.no_grad():
                outputs = self(inputs)

            self.callbacks.on_predict_batch_end(idx)

        self.callbacks.on_predict_end()

        return outputs
