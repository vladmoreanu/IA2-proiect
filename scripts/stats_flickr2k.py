from datasets import Flickr2K
from modeling.metrics import PSNR, MSE

from pathlib import Path
from multiprocessing import freeze_support

from torch.utils.data import DataLoader
from tqdm import tqdm

from .cache_flickr2k import DATASET_ARGS


DATALOADER_ARGS = dict(
    batch_size=2,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=True,
    persistent_workers=True,
)


def iterate(i: int):
    log = {}
    dataset = Flickr2K(subset="pairs", **DATASET_ARGS[i])

    loader = DataLoader(dataset, **DATALOADER_ARGS)

    psnr = PSNR()
    mse = MSE()

    psnr.reset()
    mse.reset()
    for idx, (noisy, clean) in enumerate(tqdm(loader)):
        psnr.update(clean, noisy)
        mse.update(clean, noisy)
        # if idx > 15:
        #     break

    log["psnr"] = psnr.result()
    log["mse"] = mse.result()
    return log


def main():
    path = Path("results/dataset_stats.csv")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(",".join(["Index", "PSNR", "MSE"]))
        fp.write("\n")
    for i in range(len(DATASET_ARGS)):
        log = iterate(i)
        with open(path, "a", encoding="utf-8") as fp:
            fp.write(",".join([str(x) for x in [i, log["psnr"], log["mse"]]]))
            fp.write("\n")


if __name__ == "__main__":
    freeze_support()
    main()
