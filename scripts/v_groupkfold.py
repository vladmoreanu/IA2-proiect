from models import DnCNN
from datasets import Flickr2K

# import lighter

# import toml

import torch
from torch.utils.data import DataLoader, RandomSampler

# from sklearn.model_selection import GroupKFold

# import pandas as pd
# import numpy as np

# import os
# from PIL import Image
# from torch.utils.data import Dataset
# from torchvision import transforms

# from datetime import datetime

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

    # # change working dir to...
    # root = '_fit/proiect/'
    # # if not os.path.exists(root):
    # #     os.makedirs(root)
    # os.chdir(root)

    config = toml.load('DnCNN_0.toml')

    dataset = Flickr2K(
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

        hist = model.fit(
            train_loader,
            validation_loader=val_loader,
            callbacks=[
                lighter.callbacks.CSVLogger(logsPath),
                lighter.callbacks.Checkpoint(
                    checkpointPath, save_best_only=True,
                ),
            ],
            **config['fit'],
        )

if __name__ == '__main__':
    main()