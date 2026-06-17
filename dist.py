#!/usr/bin/env python

import shutil
import zipfile
from pathlib import Path

EXCLUDE_DIRS = {".git", "dist", "__pycache__", ".venv", "venv", ".bin"}


def find_python_files(root: Path):
    for path in root.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


def zip_python_files(root: Path, zip_path: Path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    py_files = list(find_python_files(root))

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in py_files:
            arcname = file_path.relative_to(root)  # preserves relative path
            zf.write(file_path, arcname)

    return py_files


def copy_pdfs_from_bin(root: Path, dest: Path):
    bin_dir = root / ".bin"
    dest.mkdir(parents=True, exist_ok=True)

    if not bin_dir.exists():
        return []

    pdf_files = list(bin_dir.rglob("*.pdf"))
    for pdf_file in pdf_files:
        shutil.copy2(pdf_file, dest / pdf_file.name)

    return pdf_files


def main():
    root = Path(__file__).resolve().parent
    dist = root / "dist"

    zip_python_files(root, dist / "code.zip")
    copy_pdfs_from_bin(root, dist)


if __name__ == "__main__":
    main()