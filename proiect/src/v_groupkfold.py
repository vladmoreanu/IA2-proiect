from model import DnCNN
from dataset import ImageDataset

import lighter

import torch
from torch.utils.data import DataLoader, random_split, RandomSampler

from sklearn.model_selection import GroupKFold

import pandas as pd
import numpy as np

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

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


path_clean = r'./DATASETS/Flickr2K/normal_images_tiles'
path_noisy = r'./DATASETS/Flickr2K/noise_images_tiles'

num_of_layers = 17  # 9  # default 17

# train_split = 0.95
batch_size = 16
num_workers = 4
prefetch_factor = 6

learning_rt = 1e-3

epochs = 20

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'System :')
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  CUDA version PyTorch expects: {torch.version.cuda}')
print(f'  Using device: {device}, {torch.device(device)}')

noisy_filenames = np.array(sorted(os.listdir(path_clean)))
clean_filenames = np.array(sorted(os.listdir(path_noisy)))
groups = []
for x in clean_filenames:
    groups.append(int(x[:6]))

gkfGen = GroupKFold(n_splits=5, shuffle=False)

for fold, (train_split, val_split) in enumerate(gkfGen.split(
    noisy_filenames, clean_filenames, groups
)):
    train_dataset = ImageDataset(
        noisy_filenames[train_split],
        clean_filenames[train_split],
        path_noisy,
        path_clean
    )

    val_dataset = ImageDataset(
        noisy_filenames[val_split],
        clean_filenames[val_split],
        path_noisy,
        path_clean
    )

    # sampler = RandomSampler(train_dataset, num_samples=1600, replacement=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory_device=device,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader   = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory_device=device,
        pin_memory=True,
        persistent_workers=True
    )

    model = DnCNN(num_of_layers=num_of_layers)

    model.compile(
        torch.optim.Adam(model.parameters(), lr=learning_rt),
        torch.nn.MSELoss(),
        metrics=[lighter.metrics.PSNR()],
        device=device
    )

    checkpointPath = './checkpoints/DnCNN_{fold}.pt'.format(fold=fold)

    train_l, test_l = model.fit(
        train_loader,
        epochs=epochs,
        validation_loader=val_loader,
        validation_freq=10,
        callbacks=[
            lighter.callbacks.History(),
            lighter.callbacks.Checkpoint(
                checkpointPath, save_best_only=True,
                ),
        ]
    )

    data = { 'train_l': train_l, 'test_l': test_l }
    df = pd.DataFrame(data)
    df.to_csv('./results/DnCNN_{fold}.csv'.format(fold=fold), index=False)
