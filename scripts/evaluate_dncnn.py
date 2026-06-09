from modeling import DnCNN
from modeling.metrics import PSNR
from modeling.callbacks import Reporter
from datasets import Flickr2K
from utils import DEVICE
from utils.env import system_spec

import lighter

import json
import warnings
from pathlib import Path
from multiprocessing import freeze_support
from datetime import datetime

import typer
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold

from .cache_flickr2k import DATASET_ARGS


RESULT_ROOT = Path("./results").resolve()
CONFIG_PATH = Path("./results/DnCNN/ds_06/conf-2026-06-09_01-31.json").resolve()


app = typer.Typer()

@app.command()
def main():
    with open(CONFIG_PATH, "r", encoding='utf-8') as fp:
        base_config = lighter.Config(json.load(fp))

    root = RESULT_ROOT / base_config.name
    root.mkdir(parents=True, exist_ok=True)

    time = "2026-06-09_01-31" or datetime.now().strftime("%Y-%m-%d_%H-%M")

    warnings.filterwarnings(
        action="ignore", category=UserWarning, message="TypedStorage is deprecated"
    )
    device = DEVICE
    system_spec(device)

    ds_idx = 6
    ds_args = DATASET_ARGS[ds_idx]

    serialisable_ds_args = {
        **ds_args,
        "blur_params": ds_args["blur_params"]._asdict(),
    }

    run_config = lighter.Config(dict(base_config))
    run_config["dataset"] = {**base_config.dataset, **serialisable_ds_args}

    ds_root = root / f"ds_{ds_idx:02d}"
    ds_root.mkdir(parents=True, exist_ok=True)

    report_path = ds_root / base_config.report.format(time=time)
    conf_path   = ds_root / f"conf-{time}.json"

    # with open(conf_path, "w", encoding="utf-8") as fp:
    #     fp.write(run_config.to_json())

    print("=" * 80)
    print(f"DATASET {ds_idx}: {serialisable_ds_args}")
    print("=" * 80)

    dataset = Flickr2K(**run_config.dataset)

    gkf = GroupKFold(**run_config.validation)
    train_idx, val_idx = next(iter(gkf.split(dataset.samples, groups=dataset.groups)))

    # csv_path       = ds_root / base_config.csv_log.path.format(time=time)
    chkpoint_path  = ds_root / base_config.checkpoint.filepath.format(time=time)
    # backup_dirpath = ds_root / base_config.backup_restore.dirpath.format(time=time)

    train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True,  **run_config.dataloader)
    val_loader   = DataLoader(Subset(dataset, val_idx),   shuffle=False, **run_config.dataloader)

    model = DnCNN(**run_config.model)
    model.compile(
        torch.optim.Adam(model.parameters(), **run_config.optimizer),
        torch.nn.MSELoss(),
        metrics=[PSNR()],
        device=device,
    )

    model.load(chkpoint_path)

    hist_params = dict(
        epochs=10,
        steps=len(train_loader),
        val_steps=len(val_loader),
        val_freq=1,
    )

    model.evaluate(
        data_loader=val_loader, callbacks=[Reporter(report_path, hist_params)]
    )


if __name__ == "__main__":
    freeze_support()
    app()
