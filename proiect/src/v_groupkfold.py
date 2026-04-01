from model import DnCNN
from dataset import ImageDataset

import lighter

import toml

import torch
from torch.utils.data import DataLoader, RandomSampler

from sklearn.model_selection import GroupKFold

import pandas as pd
import numpy as np

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datetime import datetime

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


class ImageDataset(Dataset):
    def __init__(self, noisy_filenames, clean_filenames, noisy_dir, clean_dir):
        self.noisy_filenames = noisy_filenames
        self.clean_filenames = clean_filenames

        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.noisy_filenames)

    def __getitem__(self, index):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_filenames[index])
        clean_path = os.path.join(self.clean_dir, self.clean_filenames[index])

        noisy_img = Image.open(noisy_path)
        clean_img = Image.open(clean_path)

        return self.transform(noisy_img), self.transform(clean_img)

    def xval(self, generator):
        self.generator = generator

    def split(self):
        for train_split, val_split in self.generator.split(
            self.noisy_filenames, self.clean_filenames, self.groups
        ):
            train_dataset = ImageDataset(
                self.noisy_filenames[train_split],
                self.clean_filenames[train_split],
                self.path_noisy,
                self.path_clean
            )

            val_dataset = ImageDataset(
                self.noisy_filenames[val_split],
                self.clean_filenames[val_split],
                self.path_noisy,
                self.path_clean
            )

            yield train_dataset, val_dataset

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'System :')
    print(f'  PyTorch version: {torch.__version__}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
    print(f'  CUDA version PyTorch expects: {torch.version.cuda}')
    print(f'  Using device: {device}')

    path_clean = r'./DATASETS/Flickr2K/normal_images_tiles'
    path_noisy = r'./DATASETS/Flickr2K/noise_images_tiles'
    noisy_filenames = np.array(sorted(os.listdir(path_clean)))
    clean_filenames = np.array(sorted(os.listdir(path_noisy)))
    groups = []
    for x in clean_filenames:
        groups.append(int(x[:6]))

    l = len(clean_filenames) / 100
    noisy_filenames = noisy_filenames[:l]
    clean_filenames = clean_filenames[:l]

    # change working dir to...
    root = '_fit/proiect/'
    # if not os.path.exists(root):
    #     os.makedirs(root)
    os.chdir(root)

    config = toml.load('DnCNN_0.toml')

    dataset = ImageDataset(
        noisy_filenames,
        clean_filenames,
        path_noisy,
        path_clean
    )

    dataset.xval(GroupKFold(**config['cross-validation']))

    for fold, (train_dataset, val_dataset) in enumerate(dataset.split()):

        train_loader = DataLoader(train_dataset, **config['dataloader'])
        val_loader   = DataLoader(  val_dataset, **config['dataloader'])

        model = DnCNN(**config['model'])

        model.compile(
            torch.optim.Adam(model.parameters(), **config['optimizer']),
            torch.nn.CrossEntropyLoss(),
            metrics=[
                lighter.metrics.PSNR()
            ],
            device=device,
        )

        today = datetime.now().date()
        checkpointPath = 'DnCNN_0/model_{time}_f{fold}.pt'.format(
            fold=fold,
            time=today,
            )

        logsPath = 'DnCNN_0_logs_{time}_f{fold}.csv'.format(
            fold=fold,
            time=today,
            )

        train_l, test_l = model.fit(
            train_loader,
            validation_loader=val_loader,
            callbacks=[
                csvLogger(logsPath),
                lighter.callbacks.History(),
                lighter.callbacks.Checkpoint(
                    checkpointPath, save_best_only=True,
                    ),
            ],
            **config['fit'],
        )

if __name__ == '__main__':
    main()