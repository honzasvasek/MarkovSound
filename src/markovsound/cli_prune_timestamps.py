"""Strip pure-timestamp brackets ([2:58], [0:00-0:31], …) from the lyrics chain.

Backs up state/lyrics_chain.pkl to state/lyrics_chain.pkl.bak before writing.
Run once. The tokenizer now also drops these going forward, so they won't
come back via /lm or /understand feedback.
"""
from __future__ import annotations

import shutil
import sys
from .config import Paths
from .lyrics_markov import (
    chain_stats,
    load_lyrics_chain,
    prune_timestamps,
    save_lyrics_chain,
)


def main() -> int:
    paths = Paths.discover()
    src = paths.lyrics_chain_path
    if not src.exists():
        print(f"no lyrics chain at {src}; nothing to prune")
        return 0

    backup = src.with_suffix(src.suffix + ".bak")
    shutil.copy2(src, backup)
    print(f"backed up {src} → {backup}")

    chain, order = load_lyrics_chain(src)
    before = chain_stats(chain)
    print(f"before: {before}")

    stats = prune_timestamps(chain)
    print(f"pruned: {stats}")

    after = chain_stats(chain)
    print(f"after:  {after}")

    save_lyrics_chain(chain, order, src)
    print(f"saved → {src}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
