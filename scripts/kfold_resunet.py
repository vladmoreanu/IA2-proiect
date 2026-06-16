from modeling import ResUNet
from modeling.metrics import PSNR, MSE, Combined_L1_MSE, Combined_L1_SSIM
from modeling.callbacks import Reporter
from datasets import Flickr2K
from utils import DEVICE, subset_first_n_groups
from utils.env import system_spec

import lighter

import sys
import json
import warnings
from pathlib import Path
from multiprocessing import freeze_support
from datetime import datetime
from typing import Optional

import typer
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold

from .cache_flickr2k import DATASET_ARGS

RESULT_ROOT = Path("./results").resolve()

CONFIG = lighter.Config(
    {
        "name": "ResUNet-kfold-combined_loss_L1+SSIM",
        "report": "report-{time}.json",
        "dataset": {
            "subset": "tiled_pairs",
            "resize": 1024,
            "blur_params": {
                "kernel_size": 5,
                "kernel_sigma": 10.0,
            },
            "noise_sigma": 15.0,
            "tile_size": 128,
            "cache_batch_size": 32,
        },
        "subset_size": 100,
        "dataloader": {
            "batch_size": 16,
            "num_workers": 2,
            "prefetch_factor": 4,
            "pin_memory": True,
            "persistent_workers": True,
        },
        "validation": {
            "n_splits": 5,
        },
        "model": {
            "features": 64,
        },
        "optimizer": {"lr": 1e-4},
        "fit": {
            "epochs": 10,
            # "validation_freq": 2,
        },
        "csv_log": {"path": "logs/{time}-{fold}.csv"},
        "checkpoint": {
            "filepath": "checkpoints/{time}-{fold}.pth",
            "save_best_only": True,
        },
    }
)

app = typer.Typer()


@app.command()
def main(time: Optional[str] = typer.Argument(None), only_ds: Optional[int] = typer.Option(None, "--only-ds")):
    base_config = CONFIG
    root = RESULT_ROOT / base_config.name

    root.mkdir(parents=True, exist_ok=True)

    time = time or datetime.now().strftime("%Y-%m-%d_%H-%M")

    warnings.filterwarnings(
        action="ignore", category=UserWarning, message="TypedStorage is deprecated"
    )

    csv_fmt = base_config.csv_log.path
    chkpoint_fmt = base_config.checkpoint.filepath

    device = DEVICE
    system_spec(device)

    for ds_idx, ds_args in enumerate(DATASET_ARGS):

        if only_ds is not None and ds_idx != only_ds:
            continue

        # Flatten BlurParams into a plain dict so it serialises cleanly
        serialisable_ds_args = {
            **ds_args,
            "blur_params": ds_args["blur_params"]._asdict(),
        }

        # Build a fresh run config from the base for each dataset
        run_config = lighter.Config(dict(base_config))
        run_config["dataset"] = {**base_config.dataset, **serialisable_ds_args}

        ds_root = root / f"ds_{ds_idx:02d}"
        ds_root.mkdir(parents=True, exist_ok=True)

        report_path = ds_root / base_config.report.format(time=time)

        # Check for completed or partial runs
        completed_folds = 0
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as fp:
                report = json.load(fp)
            completed_folds = len(report["results"])
            if completed_folds == run_config.validation.n_splits:
                if "results_avg" not in report:
                    keys = report["results"][0].keys()
                    report["results_avg"] = {
                        k: sum(r[k] for r in report["results"]) / len(report["results"])
                        for k in keys
                    }
                    with open(report_path, "w", encoding="utf-8") as fp:
                        json.dump(report, fp, indent=2)
                print(f"Skipping ds_{ds_idx:02d}, already completed.")
                continue

        conf_path = ds_root / f"conf-{time}.json"
        with open(conf_path, "w", encoding="utf-8") as fp:
            fp.write(run_config.to_json())

        print("=" * 80)
        print(f"DATASET {ds_idx + 1}/{len(DATASET_ARGS)}: {serialisable_ds_args}")
        print("=" * 80)

        dataset = Flickr2K(**run_config.dataset)
        ds = subset_first_n_groups(dataset, run_config.subset_size)

        gkf = GroupKFold(**run_config.validation)

        for i, (train_idx, val_idx) in enumerate(gkf.split(ds.samples, groups=ds.groups)):
            fold = i + 1
            if fold <= completed_folds:
                continue

            log_path = ds_root / csv_fmt.format(time=time, fold=fold)
            if log_path.exists():
                print(f"  Log exists for fold {fold}, skipping.")
                continue

            print("=" * 80)
            print(f"FOLD {fold}/{run_config.validation.n_splits}")
            print("=" * 80)

            path_kwargs = dict(time=time, fold=fold)
            csv_path = ds_root / csv_fmt.format(**path_kwargs)
            chkpoint_path = ds_root / chkpoint_fmt.format(**path_kwargs)

            train_ds = Subset(ds, train_idx)
            val_ds = Subset(ds, val_idx)

            train_loader = DataLoader(train_ds, shuffle=True, **run_config.dataloader)
            val_loader = DataLoader(val_ds, shuffle=False, **run_config.dataloader)

            model = ResUNet(**run_config.model)

            model.compile(
                torch.optim.Adam(model.parameters(), **run_config.optimizer),
                Combined_L1_SSIM(),
                metrics=[PSNR(), MSE()],
                device=device,
            )

            run_config["csv_log.path", "checkpoint.filepath"] = csv_path, chkpoint_path

            hist = model.fit(
                train_loader,
                validation_loader=val_loader,
                callbacks=[
                    lighter.callbacks.CSVLogger(**run_config.csv_log),
                    lighter.callbacks.Checkpoint(**run_config.checkpoint),
                ],
                **run_config.fit,
            )

            model.load_weights(chkpoint_path)

            model.evaluate(
                data_loader=val_loader, callbacks=[Reporter(report_path, hist.params)]
            )

        with open(report_path, "r", encoding="utf-8") as fp:
            report = json.load(fp)

        keys = report["results"][0].keys()
        report["results_avg"] = {
            k: sum(r[k] for r in report["results"]) / len(report["results"])
            for k in keys
        }

        with open(report_path, "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2)


if __name__ == "__main__":
    freeze_support()
    app()
