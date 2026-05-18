from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from .codes_markov import load_codes_chain, save_codes_chain
from .config import Paths
from .lyrics_markov import load_lyrics_chain, save_lyrics_chain
from .markov import load_chain, save_chain


def _dedup(chain: dict, mode: str, cap: float) -> tuple[int, int]:
    """Returns (changed, total) transition counts."""
    changed = 0
    total = 0
    for ctx, nexts in chain.items():
        for token in list(nexts.keys()):
            total += 1
            old = nexts[token]
            if mode == "binary":
                new = 1.0
            elif mode == "cap":
                new = min(old, cap)
            elif mode == "log":
                new = math.log1p(old)
            else:
                continue
            if abs(new - old) > 1e-9:
                nexts[token] = new
                changed += 1
    return changed, total


def _peak_weight(chain: dict) -> float:
    return max((w for nexts in chain.values() for w in nexts.values()), default=0.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dedup",
        description="flatten Markov chain weights to maximize diversity",
    )
    parser.add_argument("--mode", choices=["binary", "cap", "log"], default="binary",
                        help="binary=all weights→1.0 (max diversity, resets steering); "
                             "cap=clip at value; log=log(1+w) soft compression")
    parser.add_argument("--cap", type=float, default=2.0,
                        help="cap value when --mode cap (default: 2.0)")
    parser.add_argument("--only", choices=["text", "codes", "lyrics"], action="append",
                        help="restrict to chain(s); default: all three")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would change, don't save")
    args = parser.parse_args(argv)

    paths = Paths.discover()
    targets = args.only or ["text", "codes", "lyrics"]

    jobs = []
    if "text" in targets and paths.chain_path.exists():
        jobs.append(("text  ", paths.chain_path, load_chain, save_chain))
    if "codes" in targets and paths.codes_chain_path.exists():
        jobs.append(("codes ", paths.codes_chain_path, load_codes_chain, save_codes_chain))
    if "lyrics" in targets and paths.lyrics_chain_path.exists():
        jobs.append(("lyrics", paths.lyrics_chain_path, load_lyrics_chain, save_lyrics_chain))
    if not jobs:
        print("no chains to dedup", file=sys.stderr)
        return 1

    for label, path, loader, saver in jobs:
        chain, order = loader(path)
        before_peak = _peak_weight(chain)
        changed, total = _dedup(chain, args.mode, args.cap)
        after_peak = _peak_weight(chain)
        action = "would save" if args.dry_run else "saved"
        if not args.dry_run:
            saver(chain, order, path)
        print(
            f"{label} {len(chain):>6} states  {total:>7} trans  "
            f"changed={changed}  peak {before_peak:.2f}→{after_peak:.2f}  {action}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
