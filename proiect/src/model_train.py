from model import DnCNN
from dataset import ImageDataset

import lighter

import os
import random
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split, RandomSampler, Subset


def main():
    datasets = Path(os.environ.get("DATASETS"))
    path_clean = datasets / "Flickr2K/normal_images_tiles"
    path_noisy = datasets / "Flickr2K/noise_images_tiles"

    print(path_clean)
    print(path_noisy)

    warnings.filterwarnings(
        action="ignore", category=UserWarning, message="TypedStorage is deprecated"
    )

    config = lighter.Config(
        {
            "batch_size": 16,
            "train_split": 0.99,
            # 9  # default 17
            "num_of_layers": 17,
            "epochs": 10,
            "learning_rt": 1e-3,
            # number of samples per epoch for train, None to ignore
            "num_samples_train": None,
            # number of samples per epoch for validation, None to ignore
            "num_samples_val": None,
            "validation_freq": 2,
            "num_workers": 4,
            "prefetch_factor": 6,
        }
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("System :")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA version PyTorch expects: {torch.version.cuda}")
    print(f"  Using device: {device}")

    dataset = ImageDataset(noisy_dir=path_noisy, clean_dir=path_clean)

    train_size = int(len(dataset) * config.train_split)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # same split across different runs
    )

    if config.num_samples_train is not None:
        sampler = RandomSampler(
            train_dataset, num_samples=config.num_samples_train, replacement=True
        )
    else:
        sampler = None

    if config.num_samples_val is not None:
        val_indices = random.sample(range(len(val_dataset)), config.num_samples_val)
        val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory_device=device,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory_device=device,
        pin_memory=True,
        persistent_workers=True,
    )

    model = DnCNN(num_of_layers=config.num_of_layers)

    model.compile(
        torch.optim.Adam(model.parameters(), lr=config.learning_rt),
        torch.nn.MSELoss(),
        metrics=[lighter.metrics.PSNR()],
        device=device,
    )

    train_l, test_l = model.fit(
        train_loader,
        epochs=config.epochs,
        validation_loader=val_loader,
        validation_freq=config.validation_freq,
        callbacks=[
            lighter.callbacks.History(),
            lighter.callbacks.Checkpoint("./checkpoints/DnCNN_test.pt"),
        ],
    )


if __name__ == "__main__":
    main()
