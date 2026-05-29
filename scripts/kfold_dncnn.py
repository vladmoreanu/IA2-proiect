from modeling import DnCNN
from modeling.metrics import PSNR
from datasets import Flickr2K

import lighter

import toml
from pathlib import Path
import warnings
from datetime import datetime
from multiprocessing import freeze_support

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("System :")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA version PyTorch expects: {torch.version.cuda}")
    print(f"  Using device: {device}")

    config_path = Path("_fit/proiect/DnCNN_0.toml")
    config = lighter.Config(toml.load(config_path))

    out_dir = config_path.parent
    out_name = config_path.stem

    dataset = Flickr2K("tiled_pairs", device=device)

    ds = subset_first_n_groups(dataset, 100)

    gkf = GroupKFold(**config.cross_validation)

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(ds.samples, groups=ds.groups)
    ):
        train_ds = Subset(ds, train_idx)
        val_ds = Subset(ds, val_idx)

        train_loader = DataLoader(train_ds, shuffle=True, **config.dataloader)
        val_loader = DataLoader(val_ds, shuffle=False, **config.dataloader)

        model = DnCNN(**config.model)

        model.compile(
            torch.optim.Adam(model.parameters(), **config.optimizer),
            torch.nn.MSELoss(),
            metrics=[PSNR()],
            device=device,
        )

        today = datetime.now().date()

        checkpoint_path = "model_{time}_f{fold}.pt".format(
            fold=fold,
            time=today,
        )
        checkpoint_path = str(out_dir / out_name / checkpoint_path)

        log_path = "{name}_logs_{time}_f{fold}.csv".format(
            name=out_name,
            fold=fold,
            time=today,
        )
        log_path = str(out_dir / log_path)

        hist = model.fit(
            train_loader,
            validation_loader=val_loader,
            callbacks=[
                lighter.callbacks.CSVLogger(log_path),
                lighter.callbacks.Checkpoint(
                    checkpoint_path,
                    save_best_only=True,
                ),
            ],
            **config.fit,
        )


if __name__ == "__main__":
    freeze_support()
    main()
