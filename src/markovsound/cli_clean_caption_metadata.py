"""Remove legacy metadata suffix entry edges from the text Markov chain."""
from __future__ import annotations

import argparse
import shutil
import sys

from .config import Paths
from .markov import load_chain, prune_metadata_suffix_starts, save_chain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="clean-caption-metadata",
        description="remove legacy `..., bpm, in key, time` suffix paths from the text chain",
    )
    parser.add_argument("--dry-run", action="store_true", help="show what would change without saving")
    args = parser.parse_args(argv)

    paths = Paths.discover()
    if not paths.chain_path.exists():
        print(f"no text chain at {paths.chain_path}", file=sys.stderr)
        return 1
    chain, order = load_chain(paths.chain_path)
    removed = prune_metadata_suffix_starts(chain)
    print(f"removed legacy suffix entry edges: {removed}")
    if args.dry_run:
        print("would save")
        return 0
    backup = paths.chain_path.with_suffix(paths.chain_path.suffix + ".bak")
    shutil.copy2(paths.chain_path, backup)
    save_chain(chain, order, paths.chain_path)
    print(f"backed up {paths.chain_path} → {backup}")
    print("saved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
