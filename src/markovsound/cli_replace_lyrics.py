"""String replacement tool for the persisted lyrics Markov chain."""
from __future__ import annotations

import argparse
import shutil
import sys

from .config import Paths
from .lyrics_markov import chain_stats, load_lyrics_chain, replace_string, save_lyrics_chain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="replace-lyrics",
        description="replace a string inside lyrics-chain tokens",
    )
    parser.add_argument("old", help="string to find, e.g. 'Rap' or '[Verse - Rap]'")
    parser.add_argument("new", help="replacement token")
    parser.add_argument("--dry-run", action="store_true", help="show changes without saving")
    args = parser.parse_args(argv)

    paths = Paths.discover()
    path = paths.lyrics_chain_path
    if not path.exists():
        print(f"no lyrics chain at {path}", file=sys.stderr)
        return 1
    if args.old == args.new:
        print("old and new token are identical; nothing to do")
        return 0

    chain, order = load_lyrics_chain(path)
    before = chain_stats(chain)
    stats = replace_string(chain, args.old, args.new)
    after = chain_stats(chain)
    action = "would save" if args.dry_run else "saved"
    print(f"replace {args.old!r} → {args.new!r}")
    print(f"before: {before}")
    print(f"changed: {stats}")
    print(f"after:  {after}")
    if not args.dry_run:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        save_lyrics_chain(chain, order, path)
        print(f"backed up {path} → {backup}")
    print(action)
    return 0


if __name__ == "__main__":
    sys.exit(main())
