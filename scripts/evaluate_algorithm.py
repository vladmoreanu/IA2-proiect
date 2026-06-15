import json
import warnings
from pathlib import Path
from multiprocessing import freeze_support
from datetime import datetime
import math

import typer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold

# Importuri din structura proiectului
from datasets import Flickr2K
from utils import DEVICE
from utils.env import system_spec
from .cache_flickr2k import DATASET_ARGS
import lighter

from modeling.metrics import psnr
from .richardson_lucy_algorithm import main_pipeline

RESULT_ROOT = Path("./results").resolve()
CONFIG_PATH = Path("./results/Algorithm/conf.json").resolve()

img_nr_limit = 50

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
        "blur_params": ds_args["blur_params"]._asdict() if hasattr(ds_args["blur_params"], "_asdict") else ds_args[
            "blur_params"],
    }

    run_config = lighter.Config(dict(base_config))
    run_config["dataset"] = {**base_config.dataset, **serialisable_ds_args}

    ds_root = root / f"ds_{ds_idx:02d}_classic"
    ds_root.mkdir(parents=True, exist_ok=True)
    report_path = ds_root / base_config.report.format(time=time)

    print("=" * 80)
    print(f"EVALUARE METODĂ CLASICĂ")
    print(f"DATASET {ds_idx}: {serialisable_ds_args}")
    print("=" * 80)

    dataset = Flickr2K(**run_config.dataset)
    gkf = GroupKFold(**run_config.validation)
    train_idx, val_idx = next(iter(gkf.split(dataset.samples, groups=dataset.groups)))

    val_idx = val_idx[:img_nr_limit]

    loader_args = {**run_config.dataloader, "batch_size": 1}

    val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False, **loader_args)

    blur_params = run_config.dataset["blur_params"]
    kernel_size = blur_params["kernel_size"]
    sigma = blur_params["kernel_sigma"]

    print(f"Generare kernel gaussian din config: size={kernel_size}x{kernel_size}, sigma={sigma}")

    coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel_tensor = kernel_2d.view(1, 1, kernel_size, kernel_size).to(device)

    # -------------------------------------------------------------------------
    # Evaluare
    # -------------------------------------------------------------------------
    total_psnr = 0.0
    num_batches = len(val_loader)

    print(f"Începere evaluare pe {num_batches} batch-uri...")

    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            inputs, targets = data[0].to(device), data[1].to(device)

            # Apelul funcției izolate
            outputs = main_pipeline(inputs, kernel_tensor, num_iters=30)

            batch_psnr = psnr(outputs, targets)
            total_psnr += batch_psnr

            print(f"Batch {batch_idx + 1}/{num_batches} | PSNR curent: {batch_psnr:.2f} dB")

    avg_psnr = total_psnr / num_batches
    print("=" * 80)
    print("REZULTAT FINAL VALIDARE")
    print(f"PSNR Mediu: {avg_psnr:.2f} dB")
    print("=" * 80)

    print(f"Salvat in {report_path}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Metoda: Clasic Iterativ (Bilateral + Richardson-Lucy)\n")
        f.write(f"PSNR: {avg_psnr:.4f}\n")


if __name__ == "__main__":
    freeze_support()
    app()