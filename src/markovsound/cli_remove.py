from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Paths
from .markov import load_chain, save_chain, untrain_text
from .codes_markov import load_codes_chain, save_codes_chain, untrain_codes, parse_codes
from .lyrics_markov import load_lyrics_chain, save_lyrics_chain, untrain_lyrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verwijder",
        description="remove a track (mp3 + json sidecar) and untrain its caption from the chain",
    )
    parser.add_argument("track", help="path to an MP3 file in Audio/")
    args = parser.parse_args(argv)

    paths = Paths.discover()
    track = Path(args.track).resolve()
    if not track.exists():
        print(f"not found: {track}", file=sys.stderr)
        return 1
    if track.suffix.lower() != ".mp3":
        print(f"not an mp3: {track}", file=sys.stderr)
        return 1

    sidecar = track.with_suffix(".json")
    trained_caption = None
    seed_caption = None
    audio_codes = None
    lyrics = None
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text())
            trained_caption = data.get("trained_caption")
            seed_caption = data.get("caption")
            audio_codes = data.get("audio_codes")
            lyrics = data.get("used_lyrics") or data.get("lyrics")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: could not read sidecar {sidecar.name}: {exc}", file=sys.stderr)

    # Untrain Text Chain
    if paths.chain_path.exists() and (trained_caption or seed_caption):
        chain, order = load_chain(paths.chain_path)
        removed = 0
        for text in (trained_caption, seed_caption):
            if text:
                removed += untrain_text(chain, text, order)
        save_chain(chain, order, paths.chain_path)
        print(f"untrained {removed} transition(s) from text chain")

    # Untrain Codes Chain
    if paths.codes_chain_path.exists() and audio_codes:
        c_chain, c_order = load_codes_chain(paths.codes_chain_path)
        seq = parse_codes(audio_codes)
        removed = untrain_codes(c_chain, seq, c_order)
        save_codes_chain(c_chain, c_order, paths.codes_chain_path)
        print(f"untrained {removed} transition(s) from codes chain")

    # Untrain Lyrics Chain
    if paths.lyrics_chain_path.exists() and lyrics:
        l_chain, l_order = load_lyrics_chain(paths.lyrics_chain_path)
        removed = untrain_lyrics(l_chain, lyrics, l_order)
        save_lyrics_chain(l_chain, l_order, paths.lyrics_chain_path)
        print(f"untrained {removed} transition(s) from lyrics chain")

    track.unlink()
    print(f"removed {track.name}")
    if sidecar.exists():
        sidecar.unlink()
        print(f"removed {sidecar.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
