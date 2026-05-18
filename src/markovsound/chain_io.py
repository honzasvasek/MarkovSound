"""Shared persistence helpers for the project's Markov-style chains."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_chain_payload(chain: dict, order: int, path: Path) -> None:
    """Persist a Markov-style chain using the shared on-disk envelope."""
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {context: dict(nexts) for context, nexts in chain.items()}
    with path.open("wb") as handle:
        pickle.dump({"chain": normalized, "order": order}, handle)


def load_chain_payload(path: Path) -> tuple[dict, int]:
    """Load a chain saved by :func:`save_chain_payload`."""
    with path.open("rb") as handle:
        data: dict[str, Any] = pickle.load(handle)
    return data["chain"], data["order"]
