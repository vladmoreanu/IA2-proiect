import os
from pathlib import Path

path = Path(os.environ.get("DATASET", None))
print(path.absolute())
