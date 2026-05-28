from models import DnCNN
from datasets import Flickr2K

import lighter

import warnings
from multiprocessing import freeze_support

import torch
from torch.utils.data import DataLoader, RandomSampler, Subset
from sklearn.model_selection import GroupShuffleSplit


def subset_first_n_groups(dataset, n: int) -> Subset:
    if dataset.groups is None:
        raise ValueError("Dataset does not expose groups")

    if n < 1:
        raise ValueError("n must be >= 1")

    seen = []
    selected_indices = []

    for idx, group in enumerate(dataset.groups):
        if group not in seen:
            if len(seen) >= n:
                break
            seen.append(group)

        if group in seen:
            selected_indices.append(idx)

    subset = Subset(dataset, selected_indices)
    subset.samples = [dataset.samples[i] for i in selected_indices]
    subset.groups = [dataset.groups[i] for i in selected_indices]
    return subset


def main():
    warnings.filterwarnings(
        action="ignore", category=UserWarning, message="TypedStorage is deprecated"
    )

    config = lighter.Config(
        {
            "batch_size": 16,
            "test_size": 0.2,
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
            "prefetch_factor": 8,
            "chkpoint": "chkpoint/DnCNN_today.pt",
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("System :")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA version PyTorch expects: {torch.version.cuda}")
    print(f"  Using device: {device}")

    dataset = Flickr2K("tiled_pairs", device=device)

    ds = subset_first_n_groups(dataset, 100)

    gss = GroupShuffleSplit(test_size=0.2, random_state=0)

    train_idx, val_idx = next(gss.split(ds.samples, groups=ds.groups))

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    if config.num_samples_train is not None:
        shuffle = None
        sampler = RandomSampler(
            train_ds, num_samples=config.num_samples_train, replacement=True
        )
    else:
        shuffle = True
        sampler = None

    # if config.num_samples_val is not None:
    #     val_indices = random.sample(range(len(val_ds)), config.num_samples_val)
    #     val_ds = Subset(val_ds, val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
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

    hist = model.fit(
        train_loader,
        epochs=config.epochs,
        validation_loader=val_loader,
        validation_freq=config.validation_freq,
        callbacks=[
            lighter.callbacks.Checkpoint(config.chkpoint),
        ],
    )


if __name__ == "__main__":
    freeze_support()
    main()
