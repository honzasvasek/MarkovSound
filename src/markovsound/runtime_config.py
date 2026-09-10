"""Validated live settings reloaded by the loop between cycles."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    min_duration_minutes: int = 2
    max_duration_minutes: int = 10
    caption_min_words: int = 18
    caption_max_words: int = 70
    vocal_prob: float = 0.5
    takes_per_song: int = 3
    cover_strength: float = 0.6
    target_buffer_tracks: int = 1
    latent_splice_seconds: float = 5.0
    form_arc_prob: float = 0.75

    def duration_minutes(self) -> list[int]:
        return list(range(self.min_duration_minutes, self.max_duration_minutes + 1))


def _coerce(data: dict) -> RuntimeConfig:
    cfg = RuntimeConfig(
        min_duration_minutes=int(data.get("min_duration_minutes", 2)),
        max_duration_minutes=int(data.get("max_duration_minutes", 10)),
        caption_min_words=int(data.get("caption_min_words", 18)),
        caption_max_words=int(data.get("caption_max_words", 70)),
        vocal_prob=float(data.get("vocal_prob", 0.5)),
        takes_per_song=int(data.get("takes_per_song", 3)),
        cover_strength=float(data.get("cover_strength", 0.6)),
        target_buffer_tracks=int(data.get("target_buffer_tracks", 1)),
        latent_splice_seconds=float(data.get("latent_splice_seconds", 5.0)),
        form_arc_prob=float(data.get("form_arc_prob", 0.75)),
    )
    if cfg.min_duration_minutes < 1:
        raise ValueError("min_duration_minutes must be >= 1")
    if cfg.max_duration_minutes < cfg.min_duration_minutes:
        raise ValueError("max_duration_minutes must be >= min_duration_minutes")
    if cfg.caption_min_words < 1:
        raise ValueError("caption_min_words must be >= 1")
    if cfg.caption_max_words < cfg.caption_min_words:
        raise ValueError("caption_max_words must be >= caption_min_words")
    if not 0.0 <= cfg.vocal_prob <= 1.0:
        raise ValueError("vocal_prob must be between 0.0 and 1.0")
    if cfg.takes_per_song < 1:
        raise ValueError("takes_per_song must be >= 1")
    if not 0.0 <= cfg.cover_strength <= 1.0:
        raise ValueError("cover_strength must be between 0.0 and 1.0")
    if cfg.target_buffer_tracks < 1:
        raise ValueError("target_buffer_tracks must be >= 1")
    if cfg.latent_splice_seconds < 0:
        raise ValueError("latent_splice_seconds must be >= 0")
    if not 0.0 <= cfg.form_arc_prob <= 1.0:
        raise ValueError("form_arc_prob must be between 0.0 and 1.0")
    return cfg


def load_runtime_config(path: Path) -> RuntimeConfig:
    if not path.exists():
        cfg = RuntimeConfig()
        save_runtime_config(path, cfg)
        return cfg
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid runtime config at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"invalid runtime config at {path}: expected object")
    return _coerce(data)


def save_runtime_config(path: Path, cfg: RuntimeConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2) + "\n")
