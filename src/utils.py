import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def load_config(path=None):
    p = Path(path) if path else ROOT / "config" / "settings.json"
    if not p.exists():
        raise FileNotFoundError(f"Config missing: {p}. Copy config/settings.example.json to config/settings.json.")
    with p.open(encoding="utf-8") as f:
        return json.load(f)

def project_path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
