from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import ace_client
from .codes_markov import (
    build_codes_chain,
    chain_stats,
    format_codes,
    load_codes_chain,
    parse_codes,
    save_codes_chain,
    train_codes,
)
from .config import AceConfig, Paths
from .describe import is_too_pop, metadata_caption
from .markov import build_chain, load_chain, save_chain, train_text
from .steering import load_allowed


_DEFAULT_ORDER = 1


def _collect_audio(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            for ext in ("*.mp3", "*.wav", "*.flac", "*.ogg"):
                out.extend(sorted(p.glob(ext)))
        elif p.exists():
            out.append(p)
        else:
            print(f"warning: not found: {p}", file=sys.stderr)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="absorb",
        description="run /understand on audio file(s) and train the codes chain",
    )
    parser.add_argument("paths", nargs="*", help="audio files or directories (default: Audio/absorb/)")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild chain from scratch (re-reads codes_corpus.txt)")
    parser.add_argument("--order", type=int, default=_DEFAULT_ORDER,
                        help=f"chain order (default {_DEFAULT_ORDER})")
    args = parser.parse_args(argv)

    paths = Paths.discover()
    cfg = AceConfig.discover()
    paths.absorb_dir.mkdir(parents=True, exist_ok=True)

    targets = _collect_audio([Path(p) for p in args.paths] if args.paths else [paths.absorb_dir])
    if not targets and not args.rebuild:
        print(f"no audio in {paths.absorb_dir}; drop some files there or pass paths", file=sys.stderr)
        return 1

    if paths.codes_chain_path.exists():
        chain, order = load_codes_chain(paths.codes_chain_path)
        print(f"loaded codes chain: {chain_stats(chain)} (order={order})")
    else:
        chain, order = {}, args.order
        print(f"starting new codes chain (order={order})")

    text_chain: dict | None = None
    text_order: int = 2
    if paths.chain_path.exists():
        text_chain, text_order = load_chain(paths.chain_path)
        print(f"loaded text chain: {len(text_chain)} states (order={text_order})")
    elif paths.corpus_path.exists():
        corpus = [l.strip() for l in paths.corpus_path.read_text().splitlines() if l.strip()]
        text_chain = build_chain(corpus, text_order)
        save_chain(text_chain, text_order, paths.chain_path)
        print(f"built text chain from corpus: {len(text_chain)} states")

    if args.rebuild:
        sequences: list[list[int]] = []
        if paths.codes_corpus_path.exists():
            for line in paths.codes_corpus_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # line format: "track_name\tcomma,separated,codes"
                parts = line.split("\t", 1)
                codes_str = parts[1] if len(parts) == 2 else parts[0]
                seq = parse_codes(codes_str)
                if seq:
                    sequences.append(seq)
        chain = build_codes_chain(sequences, order=order)
        print(f"rebuilt chain from {len(sequences)} sequence(s): {chain_stats(chain)}")

    if not targets:
        save_codes_chain(chain, order, paths.codes_chain_path)
        return 0

    proc = ace_client.ensure_server(cfg)
    try:
        for audio in targets:
            print(f"absorbing {audio}")
            try:
                meta, _ = ace_client.understand(cfg, audio)
            except ace_client.AceError as exc:
                print(f"  failed: {exc}", file=sys.stderr)
                continue
            codes_str = meta.get("audio_codes") or ""
            seq = parse_codes(codes_str)
            if seq:
                added = train_codes(chain, seq, order=order)
                print(f"  codes  +{added} transitions (len {len(seq)})")
                with paths.codes_corpus_path.open("a") as handle:
                    handle.write(f"{audio.name}\t{format_codes(seq)}\n")
            else:
                print(f"  no audio_codes in /understand response")

            if text_chain is not None:
                trained = metadata_caption(meta)
                if not trained:
                    print(f"  text   (no usable caption)")
                elif is_too_pop(trained, allowed=load_allowed(paths.state_dir)):
                    print(f"  text   skipped (pop): {trained[:100]}...")
                else:
                    added = train_text(text_chain, trained, text_order)
                    print(f"  text   +{added} transitions: {trained[:100]}{'...' if len(trained) > 100 else ''}")

        save_codes_chain(chain, order, paths.codes_chain_path)
        print(f"saved codes chain: {chain_stats(chain)}")
        if text_chain is not None:
            save_chain(text_chain, text_order, paths.chain_path)
            print(f"saved text chain: {len(text_chain)} states")
    finally:
        ace_client.shutdown_server(proc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
