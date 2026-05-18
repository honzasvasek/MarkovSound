"""Coordination helpers between generation and the interactive player."""
from __future__ import annotations

import json
import time

from .config import Paths


def _now_playing_name(paths: Paths) -> str | None:
    try:
        return json.loads((paths.state_dir / "now_playing.json").read_text()).get("file")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def queued_tracks(paths: Paths) -> list:
    """Return unplayed tracks waiting in FIFO queue order."""
    if not paths.queue_dir.exists():
        return []
    now_playing = _now_playing_name(paths)
    return [p for p in sorted(paths.queue_dir.glob("*.mp3")) if p.name != now_playing]


def wait_for_buffer_space(
    paths: Paths,
    target_buffer_tracks: int,
    stop_flag: dict | None = None,
    log=print,
) -> None:
    """Block while enough *unplayed* tracks are already queued.

    A track leaves `Audio/queue/` as soon as `./play` claims it, so the
    currently-playing track does not count toward the future buffer. This
    lets creation of the next song begin immediately when playback starts.
    """
    logged = False
    while len(queued_tracks(paths)) >= target_buffer_tracks:
        if stop_flag and stop_flag.get("requested"):
            return
        if not logged:
            log(f"buffer full ({len(queued_tracks(paths))}/{target_buffer_tracks} queued); waiting for player")
            logged = True
        time.sleep(2)
