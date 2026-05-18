from __future__ import annotations

import json
import time
from pathlib import Path

from .config import Paths

def now_playing_path(paths: Paths) -> Path:
    return paths.state_dir / "now_playing.json"


def wait_for_player(paths: Paths, stop_flag: dict | None = None, log=print) -> None:
    """Block until the currently-playing track is expected to finish.

    `./play` writes state/now_playing.json {file, started_at, duration} each
    time it starts a track. If that file is missing or its expected end is
    already in the past (with a 60s grace for staleness), return immediately
    — no active player, so create runs at full speed. Otherwise poll-sleep
    until the track's remaining playback time has elapsed so the next
    generated slot lands in sync with the next playback turn.
    """
    state_path = now_playing_path(paths)
    logged = False
    while True:
        if stop_flag and stop_flag.get("requested"):
            return
        if not state_path.exists():
            return
        try:
            state = json.loads(state_path.read_text())
            started = float(state.get("started_at", 0))
            duration = float(state.get("duration", 0))
        except (OSError, ValueError, TypeError):
            return
        expected_end = started + duration
        now = time.time()
        # State is stale → player is gone.
        if now > expected_end + 60:
            return
        remaining = expected_end - now
        if remaining <= 1:
            return
        if not logged:
            log(f"player active ({state.get('file', '?')}); waiting {remaining:.0f}s for it to finish")
            logged = True
        time.sleep(min(remaining, 5))


