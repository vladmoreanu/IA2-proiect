import json
import csv
import typer
from pathlib import Path
import matplotlib.pyplot as plt


RESULT_ROOT = Path("./results")


def load_dataset_stats(result_root: Path) -> dict:
    stats = {}
    with open(result_root / "dataset_stats.csv", "r", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            stats[int(row["index"])] = {
                "psnr": float(row["psnr"]),
                "mse": float(row["mse"]),
            }
    return stats


def plot_folds(ds_idx: int, folds_data: list[dict], stats: dict, ds_root: Path, time: str):
    for metric, ylabel, stat_key, stat_fmt, filename in [
        ("psnr", "PSNR (dB)", "psnr", lambda v: f"Baseline {v:.1f} dB", f"plot_psnr-{time}.png"),
        ("loss", "Loss",      "mse",  lambda v: f"Baseline {v:.1e}",     f"plot_loss-{time}.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for fold, data in enumerate(folds_data, 1):
            epochs = range(1, len(data[metric]) + 1)
            ax.plot(epochs, data[metric], label=f"Fold {fold}")
        if ds_idx in stats:
            v = stats[ds_idx][stat_key]
            ax.axhline(v, color="red", linestyle="--", label=stat_fmt(v))
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Epoch")
        if min(v for data in folds_data for v in data[metric]) < 0.1:
            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.legend()
        plt.tight_layout()
        fig.savefig(ds_root / filename, dpi=600)
        plt.close(fig)


def main(time: str, experiment: str = "DnCNN-kfold-composed"):
    root = RESULT_ROOT / experiment
    stats = load_dataset_stats(RESULT_ROOT)

    summary_rows = []

    for ds_path in sorted(root.glob("ds_*")):
        ds_idx = int(ds_path.name.split("_")[1])

        # --- summary CSV ---
        report_path = ds_path / f"report-{time}.json"
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as fp:
                report = json.load(fp)
            if "results_avg" in report:
                avg = report["results_avg"]
                summary_rows.append((ds_idx, avg["val_psnr"], avg["val_loss"]))
            else:
                print(f"No results_avg in {report_path}, skipping from summary.")
        else:
            print(f"Missing report: {report_path}, skipping from summary.")

        # --- plots ---
        folds_data = []
        is_kfold = "kfold" in experiment
        log_paths = (
            [ds_path / f"logs/{time}-{fold}.csv" for fold in range(1, 6)]
            if is_kfold
            else [ds_path / f"logs/{time}.csv"]
        )
        for log_path in log_paths:
            if not log_path.exists():
                break
            psnr_vals, loss_vals = [], []
            with open(log_path, "r", encoding="utf-8") as fp:
                for row in csv.DictReader(fp):
                    psnr_vals.append(float(row["val_psnr"]))
                    loss_vals.append(float(row["val_loss"]))
            folds_data.append({"psnr": psnr_vals, "loss": loss_vals})

        if folds_data:
            plot_folds(ds_idx, folds_data, stats, ds_path, time)
            print(f"Saved plot for ds_{ds_idx:02d}")
        else:
            print(f"No log files found for ds_{ds_idx:02d}, skipping plot.")

    if summary_rows:
        summary_rows.sort()
        summary_path = root / f"summary_{time}.csv"
        with open(summary_path, "w", encoding="utf-8") as fp:
            fp.write("INDEX,AVG_PSNR,AVG_MSE\n")
            for row in summary_rows:
                fp.write(",".join(str(v) for v in row) + "\n")
        print(f"Saved {len(summary_rows)} rows to {summary_path}")
    else:
        print("No completed reports found.")


if __name__ == "__main__":
    typer.run(main)
    