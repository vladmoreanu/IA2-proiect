from datasets.flickr2k.fetch import ensure_fetched

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help='Downloads Flick2K')

@app.command()
def fetch(
    directory: Optional[Path] = typer.Option(
        None,
        "--directory",
        "-d",
        help="Base datasets directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
    ),
    zipfile: Optional[Path] = typer.Option(
        None,
        "--zipfile",
        "-z",
        help="Path to store the downloaded ZIP file",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-download and re-extract even if already present",
    ),
):
    """
    Lazily download and extract the Flickr2K dataset.
    """
    dataset_path = ensure_fetched(
        directory=directory,
        zipfile=zipfile,
        force=force,
    )

    typer.echo(f"Flickr2K dataset ready at: {dataset_path}")

if __name__ == "__main__":
    app()