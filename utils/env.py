import os
from pathlib import Path
from typing import Optional

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resolve_datasets_dir(override: Optional[Path]) -> Path:
    datasets_env = os.environ.get("DATASETS")
    if override is None and datasets_env is None:
        raise ValueError(
            "No dataset directory provided and DATASETS env var is not set."
        )
    return (override or Path(datasets_env)).expanduser().resolve()