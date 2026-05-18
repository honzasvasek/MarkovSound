"""Append-only lyric audit corpora for requested and heard lyrics."""
from __future__ import annotations

import time
from pathlib import Path

from .config import Paths

def append_lyrics_corpus(
    out_path: Path,
    *,
    lyrics: str,
    cycle: int | str,
    mp3_name: str,
    duration: float | int | None,
    bpm,
    key,
    ts,
    lang,
    vocal_mode: bool,
    lyrics_source: str,
) -> None:
    """Append one lyrics block to a corpus file with a tagged header.

    Header format is consistent across understood_lyrics.txt and
    used_lyrics.txt so they can be diffed / joined cycle-by-cycle.
    """
    text = (lyrics or "").rstrip()
    if not text:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    header = (
        f"=== {stamp} cycle={cycle} file={mp3_name} dur={duration}s "
        f"lang={lang or '?'} bpm={bpm if bpm is not None else '?'} "
        f"key={key or '?'} ts={ts or '?'}/4 "
        f"vocal_mode={vocal_mode} src={lyrics_source} ==="
    )
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(header + "\n")
        fh.write(text + "\n\n")


def append_understood_lyrics(
    paths: Paths,
    *,
    cycle: int | str,
    mp3_name: str,
    meta: dict,
    duration: float | int | None,
    vocal_mode: bool,
    lyrics_source: str,
) -> None:
    """Append the heard lyrics from this cycle to state/understood_lyrics.txt.

    Builds a flat, searchable corpus of everything ACE has sung — independent
    of the chain, so we can redesign the lyrics pipeline against real data.
    """
    append_lyrics_corpus(
        paths.lyrics_corpus_path,
        lyrics=meta.get("lyrics") or "",
        cycle=cycle,
        mp3_name=mp3_name,
        duration=duration,
        bpm=meta.get("bpm"),
        key=meta.get("keyscale"),
        ts=meta.get("timesignature"),
        lang=meta.get("vocal_language"),
        vocal_mode=vocal_mode,
        lyrics_source=lyrics_source,
    )


def append_used_lyrics(
    paths: Paths,
    *,
    cycle: int | str,
    mp3_name: str,
    enriched: dict,
    duration: float | int | None,
    vocal_mode: bool,
    lyrics_source: str,
) -> None:
    """Append the lyrics that were actually sent to /synth this cycle to
    state/used_lyrics.txt. Same header format as understood_lyrics.txt so
    the two corpora can be diffed cycle-for-cycle to see what ACE rendered
    vs what we asked it to sing."""
    append_lyrics_corpus(
        paths.used_lyrics_corpus_path,
        lyrics=enriched.get("lyrics") or "",
        cycle=cycle,
        mp3_name=mp3_name,
        duration=duration,
        bpm=enriched.get("bpm"),
        key=enriched.get("keyscale"),
        ts=enriched.get("timesignature"),
        lang=enriched.get("vocal_language"),
        vocal_mode=vocal_mode,
        lyrics_source=lyrics_source,
    )


