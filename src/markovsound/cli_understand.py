"""Standalone /understand runner.

Sends one or more audio files to ace-server's /understand endpoint and
prints the LM-transcribed lyrics + metadata. By default also appends each
result to state/understood_lyrics.txt so the corpus grows from manual
inspections too, not only from the ./create autofeedback path.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import ace_client
from .config import AceConfig, Paths
from .loop import _append_understood_lyrics


def _print_meta(meta: dict) -> None:
    keys = ("caption", "bpm", "keyscale", "timesignature", "vocal_language", "duration")
    for k in keys:
        v = meta.get(k)
        if v not in (None, "", []):
            print(f"  {k}: {v}")


def _enhance_lyrics(
    cfg: AceConfig,
    *,
    caption: str,
    vocal_language: str,
    bpm: int | float | None,
    keyscale: str | None,
    timesignature: str | None,
    duration: int | float | None,
) -> tuple[str, float]:
    """Run /lm in lm_mode='inspire' to dream lyrics that fit the
    /understand-described track. 'inspire' returns metadata + lyrics (no
    codes), so it's cheap and skips the expensive codes-generation step.
    Returns (lyrics, seconds).
    """
    request: dict = {
        "caption": caption,
        "lyrics": "",
        "vocal_language": vocal_language or "en",
        "lm_mode": "inspire",
    }
    if bpm:
        request["bpm"] = bpm
    if keyscale:
        request["keyscale"] = keyscale
    if timesignature:
        request["timesignature"] = timesignature
    if duration:
        request["duration"] = duration
    t0 = time.time()
    enriched = ace_client.lm(cfg, request)
    dt = time.time() - t0
    return (enriched.get("lyrics") or "").strip(), round(dt, 2)


def _run_one(
    cfg: AceConfig,
    paths: Paths,
    audio: Path,
    *,
    save: bool,
    json_out: bool,
    jsonl_path: Path | None = None,
    metadata: dict | None = None,
    enhance: bool = False,
) -> int:
    if not audio.exists():
        print(f"error: {audio} does not exist", file=sys.stderr)
        return 1
    print(f"\n=== {audio} ===")
    t0 = time.time()
    try:
        meta, _latent = ace_client.understand(cfg, audio)
    except ace_client.AceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    understand_secs = round(time.time() - t0, 2)
    print(f"  /understand done in {understand_secs}s")

    enhanced_lyrics = ""
    enhance_secs = 0.0
    if enhance:
        try:
            enhanced_lyrics, enhance_secs = _enhance_lyrics(
                cfg,
                caption=meta.get("caption") or "",
                vocal_language=meta.get("vocal_language") or "en",
                bpm=meta.get("bpm"),
                keyscale=meta.get("keyscale"),
                timesignature=meta.get("timesignature"),
                duration=meta.get("duration"),
            )
            print(f"  /lm inspire done in {enhance_secs}s")
        except ace_client.AceError as exc:
            print(f"  enhance failed: {exc}", file=sys.stderr)

    if json_out:
        slim = dict(meta)
        if enhanced_lyrics:
            slim["enhanced_lyrics"] = enhanced_lyrics
        print(json.dumps(slim, indent=2, ensure_ascii=False))
    else:
        _print_meta(meta)
        lyrics = (meta.get("lyrics") or "").strip()
        if lyrics:
            print(f"  --- understand lyrics ---")
            for line in lyrics.splitlines():
                print(f"  {line}")
        else:
            print("  (no understand lyrics returned)")
        if enhanced_lyrics:
            print(f"  --- enhanced lyrics (/lm inspire) ---")
            for line in enhanced_lyrics.splitlines():
                print(f"  {line}")

    if save:
        _append_understood_lyrics(
            paths,
            cycle="standalone",
            mp3_name=audio.name,
            meta=meta,
            duration=meta.get("duration"),
            vocal_mode=True,
            lyrics_source="standalone",
        )
        print(f"  appended to {paths.lyrics_corpus_path}")

    if jsonl_path is not None:
        # JSONL: keep audio_codes (recoverable but useful), add enhance result
        # and caller-supplied tags. One self-contained record per track.
        record = dict(meta)
        record["source_audio"] = str(audio)
        record["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        record["understand_seconds"] = understand_secs
        if enhanced_lyrics:
            record["enhanced_lyrics"] = enhanced_lyrics
            record["enhance_seconds"] = enhance_secs
        if metadata:
            for k, v in metadata.items():
                if v not in (None, ""):
                    record[k] = v
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  appended JSON to {jsonl_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="understand",
        description="Run ace-server /understand on one or more audio files.",
    )
    parser.add_argument("audio", nargs="+", type=Path,
                        help="audio file(s) to analyze")
    parser.add_argument("--no-save", action="store_true",
                        help="do not append to state/understood_lyrics.txt")
    parser.add_argument("--no-autostart", action="store_true",
                        help="do not auto-start ace-server if it's not running")
    parser.add_argument("--json", action="store_true",
                        help="print the full /understand JSON instead of formatted output")
    parser.add_argument("--jsonl", type=Path,
                        help="append /understand result as one JSON line to this path "
                             "(plus any --artist/--album/--title/--track metadata)")
    parser.add_argument("--artist", default=None,
                        help="ARTIST tag to fold into the JSONL record")
    parser.add_argument("--album", default=None,
                        help="ALBUM tag to fold into the JSONL record")
    parser.add_argument("--title", default=None,
                        help="TITLE tag to fold into the JSONL record")
    parser.add_argument("--track", default=None,
                        help="TRACK number tag to fold into the JSONL record")
    parser.add_argument("--enhance", action="store_true",
                        help="after /understand, run /lm lm_mode=inspire on the "
                             "understood caption to dream lyrics that fit "
                             "(saved as enhanced_lyrics)")
    args = parser.parse_args(argv)
    metadata = {
        "artist": args.artist, "album": args.album,
        "title": args.title, "track": args.track,
    }

    cfg = AceConfig.discover()
    paths = Paths.discover()

    proc = None
    if not args.no_autostart and not ace_client.server_alive(cfg):
        proc = ace_client.ensure_server(cfg)
    elif args.no_autostart and not ace_client.server_alive(cfg):
        print(f"ace-server not running at {cfg.base_url} and --no-autostart given; exiting.",
              file=sys.stderr)
        return 1

    try:
        rc = 0
        for audio in args.audio:
            r = _run_one(
                cfg, paths, audio,
                save=not args.no_save,
                json_out=args.json,
                jsonl_path=args.jsonl,
                metadata=metadata,
                enhance=args.enhance,
            )
            rc = rc or r
        return rc
    finally:
        ace_client.shutdown_server(proc)


if __name__ == "__main__":
    sys.exit(main())
