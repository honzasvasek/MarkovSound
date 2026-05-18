from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from .config import Paths
from .markov import load_chain, save_chain, train_text


_KEEP_WEIGHT = 3.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="keep",
        description="boost a track's caption(s) in the chain (human-feedback yes-vote)",
    )
    parser.add_argument("track", help="path to an MP3 file in Audio/")
    parser.add_argument("--weight", type=float, default=_KEEP_WEIGHT,
                        help=f"extra weight to train (default: {_KEEP_WEIGHT})")
    args = parser.parse_args(argv)

    paths = Paths.discover()
    track = Path(args.track).resolve()
    sidecar = track.with_suffix(".json")
    if not sidecar.exists():
        print(f"no sidecar for {track.name}", file=sys.stderr)
        return 1
    data = json.loads(sidecar.read_text())

    captions = []
    for key in ("trained_caption", "caption"):
        text = data.get(key)
        if text and text not in captions:
            captions.append(text)
    if not captions:
        print("no captions found in sidecar", file=sys.stderr)
        return 1

    if not paths.chain_path.exists():
        print(f"no chain at {paths.chain_path}", file=sys.stderr)
        return 1

    chain, order = load_chain(paths.chain_path)
    total = 0
    for text in captions:
        total += train_text(chain, text, order, weight=args.weight)
    save_chain(chain, order, paths.chain_path)
    print(f"boosted {len(captions)} caption(s), +{total} weighted transitions")

    # Archive the track to a permanent "keeps" folder
    keeps_dir = paths.audio_dir / "keeps"
    keeps_dir.mkdir(parents=True, exist_ok=True)

    cycle = data.get("cycle", 0)
    archive_name = track.stem if track.stem.isdigit() else f"{cycle:04d}"

    archive_mp3 = keeps_dir / f"{archive_name}.mp3"
    archive_json = keeps_dir / f"{archive_name}.json"

    shutil.copy2(track, archive_mp3)
    shutil.copy2(sidecar, archive_json)
    print(f"archived to {archive_mp3.name}")

    log_path = paths.state_dir / "keeps.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{track.name}\tweight={args.weight}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
