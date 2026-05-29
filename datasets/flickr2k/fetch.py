import platform
import shutil
import subprocess
import zipfile as zip
from pathlib import Path
from typing import Optional

from datasets.preprocess._base import IMAGE_EXTS

from utils.env import resolve_datasets_dir
from utils.hashing import write_timestamp_marker


def download(url, output: Path):
    if output.suffix != '.zip':
        raise ValueError("File path must include .zip extension")
    output.parent.mkdir(parents=True, exist_ok=True)

    system = platform.system().lower()
    if system == "windows":
        curl_cmd = shutil.which("curl.exe") or shutil.which("curl")
        if curl_cmd is None:
            raise RuntimeError(
                "curl.exe not found. Please install curl or ensure it is in your PATH."
            )
    else:
        curl_cmd = shutil.which("curl")
        if curl_cmd is None:
            raise RuntimeError(
                "curl not found. Please install curl using your package manager."
            )

    command = [curl_cmd, "-L", "-o", str(output), url]
    try:
        subprocess.run(command, check=True)
        print(f"Download completed: {output}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Download failed: {e}")


def unzip(zip_path: Path , output: Path):
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")
    output.mkdir(parents=True, exist_ok=True)

    try:
        with zip.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output)

        print(f"Unzipped {zip_path} to {output}")

    except zip.BadZipFile:
        raise RuntimeError("The ZIP file is corrupted or not a valid zip archive.")


def cleanup(zipfile: Path):
    if not zipfile.is_file():
        raise IsADirectoryError('Expected the name to save the zip as.')
    # IDK if it's possible for it to be missing but we can roll with it...
    zipfile.unlink(missing_ok=True)


def is_complete(output: Path) -> bool:
    return (output / "raw" / ".done").exists()


def ensure_fetched(
    zipfile: Optional[Path] = None,
    directory: Optional[Path] = None,
    force: bool = False
):
    datasets_path = resolve_datasets_dir(directory)
    flickr2k = datasets_path / "Flickr2K"
    if not force and is_complete(flickr2k):
        return flickr2k

    url = "https://www.kaggle.com/api/v1/datasets/download/daehoyang/flickr2k"

    if zipfile is None:
        zipfile = datasets_path / "tmp" / "flickr2k.zip"

    download(url, zipfile)
    unzip(zipfile, datasets_path)

    subfolder = flickr2k / "raw"
    subfolder.mkdir(parents=True, exist_ok=True)
    for item in flickr2k.iterdir():
        # don't move the target into itself
        if item == subfolder or item.suffix.lower() not in IMAGE_EXTS:
            continue
        item.rename(subfolder / item.name)

    cleanup(zipfile)
    marker = subfolder / ".done"
    write_timestamp_marker(marker)
    return flickr2k
