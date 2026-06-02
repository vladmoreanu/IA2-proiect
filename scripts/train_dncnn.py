from modeling import DnCNN
from modeling.metrics import PSNR
from modeling.callbacks import Reporter
from datasets import Flickr2K
from utils import DEVICE, subset_first_n_groups
from utils.env import system_spec

import lighter

import warnings
from pathlib import Path
from multiprocessing import freeze_support
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit


RESULT_ROOT = Path("./results").resolve()

CONFIG = lighter.Config({
        "name": "DnCNN-composed",
        "report": "report-{time}.json",
        "dataloader": {
            "batch_size": 16,
            "num_workers": 2,
            "prefetch_factor": 4,
            "pin_memory": True,
            "persistent_workers": True,
        },
        "validation": {
            "test_size": 0.2,
        },
        "model": {
            "num_of_layers": 17,
        },
        "optimizer": {
            "lr": 1e-3
        },
        "fit": {
            "epochs": 10,
            "validation_freq": 2,
        },
        "csv_log": {
            "path": "logs/{time}.csv"
        },
        "checkpoint": {
            "filepath": "checkpoints/{time}.pth",
            "save_best_only": True,
        }
    })


def main():
    config = CONFIG
    root = RESULT_ROOT / config.name

    root.mkdir(parents=True, exist_ok=True)

    time = datetime.now()

    conf_path = root / "conf-{time}.json".format(time)
    with open(conf_path, "w", encoding="utf-8") as fp:
        fp.write(config.to_json())

    warnings.filterwarnings(
        action="ignore", category=UserWarning, message="TypedStorage is deprecated"
    )

    device = DEVICE
    system_spec(device)

    dataset = Flickr2K("tiled_pairs")

    ds = subset_first_n_groups(dataset, 100)

    gss = GroupShuffleSplit(random_state=0, **config.validation)

    train_idx, val_idx = next(gss.split(ds.samples, groups=ds.groups))

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, shuffle=True, **config.dataloader)
    val_loader = DataLoader(val_ds, shuffle=False, **config.dataloader)

    model = DnCNN(**config.model)

    model.compile(
        torch.optim.Adam(model.parameters(), lr=config.learning_rt),
        torch.nn.MSELoss(),
        metrics=[PSNR()],
        device=device,
    )

    path_kwargs = dict(
        time=time
    )

    csv_path = root / config.csv_log.path.format(**path_kwargs)
    chkpoint_path = root / config.checkpoint.filepath.format(**path_kwargs)
    report_path = root / config.report.format(**path_kwargs)

    config["csv_log.path", "checkpoint.filepath"] = csv_path, chkpoint_path

    hist = model.fit(
        train_loader=train_loader,
        validation_loader=val_loader,
        callbacks=[
            lighter.callbacks.CSVLogger(**config.csv_log),
            lighter.callbacks.Checkpoint(**config.checkpoint),
        ],
        **config.fit
    )

    model.load(chkpoint_path)

    model.evaluate(
        data_loader=val_loader,
        callbacks=[
            Reporter(report_path, hist.params)
        ]
    )

if __name__ == "__main__":
    freeze_support()
    main()
