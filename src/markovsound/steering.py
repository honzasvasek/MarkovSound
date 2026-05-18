from __future__ import annotations

import json
from pathlib import Path


def load_allowed(state_dir: Path) -> set[str]:
    path = state_dir / "steering.json"
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    return set(data.get("allowed_pop_terms", []))


def save_allowed(state_dir: Path, allowed: set[str]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "steering.json"
    path.write_text(json.dumps({"allowed_pop_terms": sorted(allowed)}, indent=2) + "\n")
