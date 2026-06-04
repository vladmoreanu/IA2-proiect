from utils import DEVICE, system_spec
from datasets.preprocess.workers import BlurParams
from datasets import Flickr2K

from multiprocessing import freeze_support

DATASET_ARGS = [
    dict(
        blur_params=BlurParams(kernel_size=5, kernel_sigma=5.0),
        noise_sigma=0.0,
    ),
    dict(
        blur_params=BlurParams(kernel_size=5, kernel_sigma=10.0),
        noise_sigma=0.0,
    ),
    dict(
        blur_params=BlurParams(kernel_size=5, kernel_sigma=15.0),
        noise_sigma=0.0,
    ),
    dict(
        blur_params=BlurParams(kernel_size=5, kernel_sigma=0.0),
        noise_sigma=15.0,
    ),
    dict(
        blur_params=BlurParams(kernel_size=5, kernel_sigma=0.0),
        noise_sigma=20.0,
    ),
    dict(
        blur_params=BlurParams(kernel_size=5, kernel_sigma=0.0),
        noise_sigma=25.0,
    ),
    dict(
        blur_params=BlurParams(kernel_size=5, kernel_sigma=10.0),
        noise_sigma=15.0,
    ),
]


def main():
    device = DEVICE
    system_spec(device)

    for i in range(len(DATASET_ARGS)):
        Flickr2K(subset="tiled_pairs", **DATASET_ARGS[i])


if __name__ == "__main__":
    freeze_support()
    main()
