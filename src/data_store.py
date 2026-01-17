import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data")
HIST_DIR = DATA_DIR / "history"
HIST_DIR.mkdir(parents=True, exist_ok=True)

def save_json(obj: dict, name: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = HIST_DIR / f"{name}_{ts}.json"
    path.write_text(json.dumps(obj, indent=2))
    return path

def load_latest(prefix: str):
    files = sorted(HIST_DIR.glob(f"{prefix}_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())
