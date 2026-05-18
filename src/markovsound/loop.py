"""Top-level loop orchestration: lifecycle, signals, live config reloads, and cycle scheduling."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import random
import re
import signal
import sys
import time
from pathlib import Path

from . import ace_client
from .config import AceConfig, Paths
from .codes_markov import chain_stats as codes_chain_stats, load_codes_chain, save_codes_chain
from .lyrics_markov import chain_stats as lyrics_chain_stats, load_lyrics_chain, save_lyrics_chain
from .markov import build_chain, load_chain, save_chain
from .cycle import (
    apply_feedback,
    plan_song,
    prepare_lyrics,
    publish_output,
    synthesize,
    train_generated,
    write_output,
)
from .playback import wait_for_buffer_space
from .runtime_config import RuntimeConfig, load_runtime_config


# Below this many transitions the codes chain isn't varied enough yet —
# fall back to /lm. With order-1 over ~64k vocab a single /understand call
# adds ~150 transitions; 1500 ≈ 10 absorbed/generated tracks.
_CODES_ORDER = 3
_LYRICS_ORDER = 3


_DEFAULT_ORDER = 3
_DEFAULT_DURATION = 0  # 0 = random duration each cycle


def log(msg: str) -> None:
    """Emit one timestamped loop status line and flush immediately."""
    timestamp = time.strftime("%H:%M:%S")
    sys.stdout.write(f"[{timestamp}] {msg}\n")
    sys.stdout.flush()


def _random_duration_seconds(cfg: RuntimeConfig) -> int:
    minutes = cfg.duration_minutes()
    # Preserve a short-track bias while allowing the range to change live.
    weights = list(range(len(minutes), 0, -1))
    return random.choices(minutes, weights=weights, k=1)[0] * 60



@dataclass
class ChainWatch:
    """Track chain file mtimes so external maintenance edits can be hot-reloaded."""
    path: Path
    mtime_ns: int | None

    @classmethod
    def from_path(cls, path: Path) -> "ChainWatch":
        return cls(path, path.stat().st_mtime_ns if path.exists() else None)

    def changed_externally(self) -> bool:
        current = self.path.stat().st_mtime_ns if self.path.exists() else None
        return current != self.mtime_ns

    def mark_current(self) -> None:
        self.mtime_ns = self.path.stat().st_mtime_ns if self.path.exists() else None


def _reload_changed_chains(
    *, paths: Paths, chain: dict, order: int, codes_chain: dict, codes_order: int,
    lyrics_chain: dict, lyrics_order: int, watches: dict[str, ChainWatch], log=print,
) -> tuple[dict, int, dict, int, dict, int]:
    """Hot-reload chain files modified by tools outside the running loop."""
    if watches["text"].changed_externally() and paths.chain_path.exists():
        chain, order = load_chain(paths.chain_path)
        watches["text"].mark_current()
        log(f"reloaded text chain from disk: {len(chain)} states, order={order}")
    if watches["codes"].changed_externally() and paths.codes_chain_path.exists():
        codes_chain, codes_order = load_codes_chain(paths.codes_chain_path)
        watches["codes"].mark_current()
        log(f"reloaded codes chain from disk: {codes_chain_stats(codes_chain)}")
    if watches["lyrics"].changed_externally() and paths.lyrics_chain_path.exists():
        lyrics_chain, lyrics_order = load_lyrics_chain(paths.lyrics_chain_path)
        watches["lyrics"].mark_current()
        log(f"reloaded lyrics chain from disk: {lyrics_chain_stats(lyrics_chain)}")
    return chain, order, codes_chain, codes_order, lyrics_chain, lyrics_order

def _load_or_build_chain(paths: Paths, order: int) -> tuple[dict, int]:
    if paths.chain_path.exists():
        chain, loaded_order = load_chain(paths.chain_path)
        log(f"loaded chain: {len(chain)} states, order={loaded_order}")
        return chain, loaded_order
    if not paths.corpus_path.exists():
        raise SystemExit(f"no chain and no corpus at {paths.corpus_path}")
    corpus = [
        line.strip() for line in paths.corpus_path.read_text().splitlines() if line.strip()
    ]
    log(f"building chain from {len(corpus)} corpus lines (order={order})")
    chain = build_chain(corpus, order)
    save_chain(chain, order, paths.chain_path)
    log(f"saved chain: {len(chain)} states")
    return chain, order


def _load_codes_chain(paths: Paths) -> tuple[dict, int]:
    if paths.codes_chain_path.exists():
        chain, order = load_codes_chain(paths.codes_chain_path)
        stats = codes_chain_stats(chain)
        log(f"loaded codes chain: {stats}")
        return chain, order
    log(f"no codes chain yet (will build as we go)")
    return {}, _CODES_ORDER


def _load_lyrics_chain(paths: Paths) -> tuple[dict, int]:
    if paths.lyrics_chain_path.exists():
        chain, order = load_lyrics_chain(paths.lyrics_chain_path)
        stats = lyrics_chain_stats(chain)
        log(f"loaded lyrics chain: {stats}")
        return chain, order
    log(f"no lyrics chain yet (will build as we go)")
    return {}, _LYRICS_ORDER


def _read_cycle(paths: Paths) -> int:
    try:
        return int(paths.cycle_path.read_text().strip())
    except (OSError, ValueError):
        return 0


def _write_cycle(paths: Paths, cycle: int) -> None:
    paths.cycle_path.parent.mkdir(parents=True, exist_ok=True)
    paths.cycle_path.write_text(f"{cycle}\n")


def _read_preset(paths: Paths) -> str:
    """Optional caption preset prepended to every fresh markov caption.

    Stored at state/preset.txt; re-read each cycle so it can be edited
    live without restarting ./create. Empty file or missing file = no
    preset.
    """
    try:
        text = paths.preset_path.read_text().strip()
    except (OSError, FileNotFoundError):
        return ""
    return text


def _run_cycle(
    cycle: int,
    chain: dict,
    order: int,
    codes_chain: dict,
    codes_order: int,
    lyrics_chain: dict,
    lyrics_order: int,
    paths: Paths,
    ace_cfg: AceConfig,
    runtime_cfg: RuntimeConfig,
    duration: int,  # 0 = pick randomly per cycle
    autofeedback: bool,
    force_codes_mode: str = "auto",  # "auto" | "markov" | "lm"
    vocal_prob: float = 0.0,
    vocal_lang: str = "en",
    force_lyrics_mode: str = "auto",  # "auto" | "markov" | "lm"
    song_state: dict | None = None,
    takes_per_song: int = 1,
    chain_watches: dict[str, ChainWatch] | None = None,
) -> bool:
    log(f"=== cycle {cycle} ===")
    plan = plan_song(
        chain=chain, order=order, paths=paths, runtime_cfg=runtime_cfg,
        requested_duration=duration, song_state=song_state, takes_per_song=takes_per_song,
        random_duration_seconds=_random_duration_seconds,
        read_preset=_read_preset, log=log,
    )
    if plan is None:
        return False

    lyrics_plan = prepare_lyrics(
        plan=plan, lyrics_chain=lyrics_chain, lyrics_order=lyrics_order,
        ace_cfg=ace_cfg, vocal_lang=vocal_lang, log=log,
    )
    result = synthesize(
        plan=plan, lyrics_plan=lyrics_plan, codes_chain=codes_chain,
        codes_order=codes_order, ace_cfg=ace_cfg, force_codes_mode=force_codes_mode,
        song_state=song_state, latent_splice_seconds=runtime_cfg.latent_splice_seconds, log=log,
    )
    mp3_path, json_path, sidecar = write_output(
        cycle=cycle, plan=plan, lyrics_plan=lyrics_plan, result=result,
        paths=paths, song_state=song_state,
    )
    train_generated(
        result=result, lyrics_plan=lyrics_plan, sidecar=sidecar,
        codes_chain=codes_chain, codes_order=codes_order,
        lyrics_chain=lyrics_chain, lyrics_order=lyrics_order,
        vocal_mode=plan.vocal_mode, log=log,
    )
    if autofeedback:
        try:
            apply_feedback(
                ace_cfg=ace_cfg, mp3_path=mp3_path, json_path=json_path, sidecar=sidecar,
                chain=chain, order=order, codes_chain=codes_chain, codes_order=codes_order,
                lyrics_chain=lyrics_chain, lyrics_order=lyrics_order, paths=paths, cycle=cycle,
                plan=plan, result=result, song_state=song_state, log=log,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"autofeedback failed: {exc!r}")
    mp3_path, json_path = publish_output(
        mp3_path=mp3_path, json_path=json_path, sidecar=sidecar, paths=paths,
    )
    if song_state is not None:
        song_state["last_mp3_path"] = str(mp3_path)
    log(f"queued {mp3_path.name}")
    save_codes_chain(codes_chain, codes_order, paths.codes_chain_path)
    save_lyrics_chain(lyrics_chain, lyrics_order, paths.lyrics_chain_path)
    if chain_watches is not None:
        # Text may have been saved by feedback; all three watches should now
        # represent this process's own latest writes, not external edits.
        for watch in chain_watches.values():
            watch.mark_current()
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="markovsound", description="generative music loop")
    parser.add_argument("-d", "--duration", type=int, default=_DEFAULT_DURATION,
                        help="fixed track duration in seconds; 0 = use live state/runtime.json duration range")
    parser.add_argument("-n", "--cycles", type=int, default=0,
                        help="stop after N cycles (0 = run forever)")
    parser.add_argument("--no-autofeedback", action="store_true",
                        help="skip /understand training step")
    parser.add_argument("--no-autostart", action="store_true",
                        help="do not auto-start ace-server")
    parser.add_argument("--order", type=int, default=_DEFAULT_ORDER,
                        help="markov order when building from corpus")
    parser.add_argument("--codes-mode", choices=["auto", "markov", "lm"], default="auto",
                        help="audio_codes source: 'auto' = markov once chain has enough, "
                             "'markov' = always force markov (errors if empty), 'lm' = always /lm")
    parser.add_argument("--vocal-lang", default="en",
                        help="ISO 639-1 vocal language (en, fr, de, nl, es, it, ja, ...)")
    parser.add_argument("--lyrics-mode", choices=["auto", "markov", "lm"], default="auto",
                        help="lyrics source on vocal cycles: 'auto' = markov once chain ready, "
                             "with /lm fallback if no sample contains a vocal section; "
                             "'markov' = always force chain (errors if empty); 'lm' = always /lm")
    parser.add_argument("--no-wait-for-player", action="store_true",
                        help="don't pause between cycles when ./play is running — "
                             "go full speed so the chain evolves faster")
    args = parser.parse_args(argv)

    paths = Paths.discover()
    ace_cfg = AceConfig.discover()
    log(f"paths: state={paths.state_dir} audio={paths.audio_dir}")
    log(f"ace: {ace_cfg.base_url} models={ace_cfg.models_dir}")

    chain, order = _load_or_build_chain(paths, args.order)
    codes_chain, codes_order = _load_codes_chain(paths)
    lyrics_chain, lyrics_order = _load_lyrics_chain(paths)
    chain_watches = {
        "text": ChainWatch.from_path(paths.chain_path),
        "codes": ChainWatch.from_path(paths.codes_chain_path),
        "lyrics": ChainWatch.from_path(paths.lyrics_chain_path),
    }

    stop_flag = {"requested": False}

    def _handle_sigint(signum, frame):  # noqa: ARG001
        # async-signal-safe: write directly, no logger lock
        os.write(2, b"\n[signal] SIGINT received; will stop after current cycle.\n")
        if stop_flag["requested"]:
            os.write(2, b"[signal] second SIGINT; exiting immediately.\n")
            os._exit(130)
        stop_flag["requested"] = True

    signal.signal(signal.SIGINT, _handle_sigint)

    proc = None
    if not args.no_autostart:
        proc = ace_client.ensure_server(ace_cfg, log=log)
    elif not ace_client.server_alive(ace_cfg):
        log(f"ace-server not running at {ace_cfg.base_url} and --no-autostart given; exiting.")
        return 1

    try:
        cycle = _read_cycle(paths)
        completed = 0
        try:
            runtime_cfg = load_runtime_config(paths.runtime_config_path)
        except ValueError as exc:
            runtime_cfg = RuntimeConfig()
            log(f"runtime config invalid at startup; using defaults: {exc}")
        log(f"runtime config: {runtime_cfg}")
        # Persistent song state — survives across cycles within one ./create
        # invocation so multiple takes share caption+duration. Cleared
        # automatically when takes_left hits 0.
        song_state: dict = {"caption": None, "duration": 0,
                            "takes_left": 0, "take_no": 0,
                            "cover_strength": runtime_cfg.cover_strength,
                            "last_mp3_path": None,
                            "last_transcribed_lyrics": "",
                            "last_audio_codes": "",
                            "runtime_cfg": runtime_cfg}
        while True:
            cycle += 1
            try:
                chain, order, codes_chain, codes_order, lyrics_chain, lyrics_order = _reload_changed_chains(
                    paths=paths, chain=chain, order=order,
                    codes_chain=codes_chain, codes_order=codes_order,
                    lyrics_chain=lyrics_chain, lyrics_order=lyrics_order,
                    watches=chain_watches, log=log,
                )
                try:
                    runtime_cfg = load_runtime_config(paths.runtime_config_path)
                except ValueError as exc:
                    log(f"runtime config invalid; keeping previous values: {exc}")
                else:
                    if runtime_cfg != song_state.get("runtime_cfg"):
                        log(f"runtime config: {runtime_cfg}")
                        song_state["runtime_cfg"] = runtime_cfg
                    song_state["cover_strength"] = runtime_cfg.cover_strength
                ok = _run_cycle(
                    cycle, chain, order, codes_chain, codes_order,
                    lyrics_chain, lyrics_order, paths, ace_cfg, runtime_cfg,
                    duration=args.duration,
                    autofeedback=not args.no_autofeedback,
                    force_codes_mode=args.codes_mode,
                    vocal_prob=runtime_cfg.vocal_prob,
                    vocal_lang=args.vocal_lang,
                    force_lyrics_mode=args.lyrics_mode,
                    song_state=song_state,
                    takes_per_song=runtime_cfg.takes_per_song,
                    chain_watches=chain_watches,
                )
            except ace_client.AceError as exc:
                log(f"cycle {cycle} failed: {exc}")
                ok = False
            if ok:
                completed += 1
                _write_cycle(paths, cycle)
            if stop_flag["requested"]:
                log("stop requested; exiting loop.")
                break
            if args.cycles and completed >= args.cycles:
                log(f"completed {completed} cycle(s); exiting.")
                break
            # Back-pressure: keep a small FIFO buffer ready for the player,
            # but start composing again as soon as the player claims a queued
            # track. --no-wait-for-player still allows intentionally unbounded
            # generation for training runs.
            if not args.no_wait_for_player:
                wait_for_buffer_space(
                    paths, runtime_cfg.target_buffer_tracks,
                    stop_flag=stop_flag, log=log,
                )
        return 0
    finally:
        ace_client.shutdown_server(proc, log=log)


if __name__ == "__main__":
    sys.exit(main())
