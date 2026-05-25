from pathlib import Path
from datetime import datetime, timezone

def write_timestamp_marker(marker: Path):
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(datetime.now(timezone.utc).isoformat())