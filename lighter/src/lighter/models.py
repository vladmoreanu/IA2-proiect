from .metrics import Metric, Loss
from .callbacks import CallbackList, History, PBar

import torch

import os

from typing import List

class Model(torch.nn.Module):
    history: History

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
        self.load_state_dict(
            torch.load(filepath, map_location=self.device, weights_only=True)
        )

    def forward(self, x):
        return x

    def compile(
        self,
        optimizer=None,
        loss=None,
        # loss_weights=None,
        metrics: List[Metric] = None,
        # weighted_metrics=None,
        # run_eagerly=False,
        # steps_per_execution=1,
        # jit_compile="auto",
        # auto_scale_loss=True,
        device="cpu",
    ):
        self.optimizer = optimizer
        self.loss_fn = loss
        self.metrics = [Loss()] + metrics
        self.device = device

        self.to(self.device)

    def step(
        self,
        inputs,
        targets,
        training: bool = False,
    ):
        pfx = "train_" if training else "val_"

        if training:
            outputs = self(inputs)
            loss = self.loss_fn(outputs, targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
        else:
            with torch.no_grad():
                outputs = self(inputs)
                loss = self.loss_fn(outputs, targets)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update(loss.item())
            else:
                metric.update(targets, outputs)

        return {pfx + x.name: x.result() for x in self.metrics}

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
        initial_epoch=1,
        restore_batch=0,
        # steps_per_epoch=None,
        # validation_rate=None,
        # validation_batch_size=None,
        validation_freq=1,
    ):
        self.to(self.device)


        self.callbacks = CallbackList(
            callbacks=(callbacks or []) + [
                History(),
                PBar(initial_batch=restore_batch)
            ],
            model=self,
            epochs=epochs,
            steps=len(train_loader),
            val_steps=len(validation_loader),
            val_freq=validation_freq,
        )

        self.callbacks.on_train_begin()

        for e in range(initial_epoch, epochs + 1):
            self.callbacks.on_epoch_begin(e)

            # Training loop
            self.train()
            for metric in self.metrics:
                metric.reset()

            _skip = restore_batch if e == initial_epoch else 0

            iterator = iter(train_loader)
            for idx in range(len(train_loader)):
                if idx < _skip:
                    continue
                inputs, targets = next(iterator)
                self.callbacks.on_train_batch_begin(idx)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                batch_log = self.step(inputs, targets, training=True)  
                self.callbacks.on_train_batch_end(idx, batch_log)

            epoch_log = batch_log

            # Validation loop
            if (validation_loader is not None) and (e % validation_freq == 0):
                self.eval()
                for metric in self.metrics:
                    metric.reset()

                for idx, (inputs, targets) in enumerate(validation_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    self.callbacks.on_val_batch_begin(idx)
                    batch_log = self.step(inputs, targets, training=False)  
                    self.callbacks.on_val_batch_end(idx, batch_log)
                
                epoch_log |= batch_log

            self.callbacks.on_epoch_end(e, epoch_log)

        self.callbacks.on_train_end(epoch_log)

        return self.history

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
            callbacks=(callbacks or []) + [History(), PBar()],
            model=self,
            steps=len(data_loader),
        )

        self.callbacks.on_val_begin()

        self.eval()
        for metric in self.metrics:
            metric.reset()

        for idx, (inputs, targets) in enumerate(validation_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.callbacks.on_val_batch_begin(idx)
            log = self.step(inputs, targets)  
            self.callbacks.on_val_batch_end(idx, log)

        self.callbacks.on_val_end(log)

        results = []
        for val in log.values():
            results.append(val)

        return results

    def predict(
        self,
        data_loader,
        # batch_size=None,
        # verbose="auto",
        # steps=None,
        callbacks=None,
    ):
        self.to(self.device)

        self.callbacks = CallbackList(
            callbacks=(callbacks or []) + [History(), PBar()],
            model=self,
            steps=len(data_loader),
        )

        self.eval()
        self.callbacks.on_predict_begin()

        all_outputs = []
        for idx, inputs in enumerate(data_loader):
            inputs = inputs.to(self.device)

            self.callbacks.on_predict_batch_begin(idx)

            with torch.no_grad():
                outputs = self(inputs)

            self.callbacks.on_predict_batch_end(idx)
            all_outputs.append(outputs.cpu())

        self.callbacks.on_predict_end()

        return torch.cat(all_outputs)

