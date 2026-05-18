from __future__ import annotations
import argparse
import sys
from pathlib import Path
from .config import Paths
from .lyrics_markov import load_lyrics_chain, save_lyrics_chain, train_lyrics

def main():
    parser = argparse.ArgumentParser(description="Seed the lyrics Markov chain")
    parser.add_argument("lyrics", help="Lyrics to train into the chain")
    args = parser.parse_args()

    paths = Paths.discover()

    # Load or create chain
    if paths.lyrics_chain_path.exists():
        chain, order = load_lyrics_chain(paths.lyrics_chain_path)
    else:
        chain = {}
        order = 2 # Default order for lyrics

    added = train_lyrics(chain, args.lyrics, order)
    save_lyrics_chain(chain, order, paths.lyrics_chain_path)
    print(f"Trained {added} transitions into the lyrics chain.")

if __name__ == "__main__":
    sys.exit(main())
