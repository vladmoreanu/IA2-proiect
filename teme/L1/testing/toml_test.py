import lighter

import torch
from torch import FloatTensor, LongTensor
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.model_selection import KFold

import os
import toml

import pandas as pd

from xor_dataset import create_xor_dataset

class myDataset(TensorDataset):
    def __init__(self, *tensors):
        super().__init__(*tensors)
        self.generator = None
        self.inputs = tensors[0]
        self.targets = tensors[-1]
        self.groups = None

    def xval(self, generator):
        self.generator = generator

    def split(self):
        for train_split, val_split in self.generator.split(
            self.inputs, self.targets, self.groups
        ):
            # THESE 4 ARE ALREADY TENSORS
            x_train, x_val = self.inputs[train_split], self.inputs[val_split]
            y_train, y_val = self.targets[train_split], self.targets[val_split]

            train_dataset = TensorDataset(
                x_train, 
                y_train
                )
            val_dataset = TensorDataset(
                x_val, 
                y_val
                )

            yield train_dataset, val_dataset


class csvLogger(lighter.callbacks.Callback):
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


class Accuracy(lighter.metrics.Metric):
    def __init__(self):
        self.name = 'acc'

    def reset(self):
        self.correct_predictions = 0
        self.total_samples = 0

    def update(self, targets, outputs):
        _, predicted = torch.max(outputs, 1)
        self.correct_predictions += (predicted == targets).sum().item()
        self.total_samples += targets.size(0)

    def result(self):
        return (self.correct_predictions / self.total_samples) \
            if self.total_samples > 0 else None


class FFN(lighter.Model):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.activation = torch.nn.Sigmoid()
        
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x


def main():
    inputs, targets = create_xor_dataset(num_samples_per_class=200, noise=0.25)

    dataset = myDataset(
        FloatTensor(inputs),
        LongTensor(targets),
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(
        f'System :'
        f'  PyTorch version: {torch.__version__}'
        f'  CUDA available: {torch.cuda.is_available()}'
        f'  CUDA version PyTorch expects: {torch.version.cuda}'
        f'  Using device: {device}, {torch.device(device)}'
    )

    config = toml.load('./_fit/xor_test/FFN_0.toml')

    dataset.xval(KFold(**config['cross-validation']))

    fold_losses = []

    for fold, (train_dataset, val_dataset) in enumerate(dataset.split()):
        train_loader = DataLoader(train_dataset, **config['dataloader'])
        val_loader   = DataLoader(  val_dataset, **config['dataloader'])

        model = FFN(**config['model'])

        model.compile(
            torch.optim.Adam(model.parameters(), **config['optimizer']),
            torch.nn.CrossEntropyLoss(),
            metrics=[
                Accuracy()
            ],
        )

        save_path = f'./_fit/xor_test/FFN_0/model_f{fold}' \
            + '_{epoch}.pt'
        train_l, test_l = model.fit(
            train_loader,
            validation_loader=val_loader,
            callbacks=[
                csvLogger(f'./_fit/xor_test/FFN_0_logs_f{fold}.csv'),
                lighter.callbacks.History(),
                lighter.callbacks.Checkpoint(
                    save_path,
                    save_best_only=True
                ),
            ],
            **config['fit'],
        )

        fold_losses.append((train_l, test_l))


if __name__ == '__main__':
    main()