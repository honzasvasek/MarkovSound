from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Paths
from .describe import steering_terms_in
from .markov import load_chain, save_chain, train_text
from .steering import load_allowed, save_allowed


def _load_preset(preset: str, prompts_dir: Path) -> list[str]:
    path = prompts_dir / "genres" / f"{preset}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"preset '{preset}' not found at {path}. "
            f"Available: {sorted(p.stem for p in (prompts_dir / 'genres').glob('*.txt'))}"
        )
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="steer",
        description="push the text chain toward a target style (high-weight train + auto-whitelist)",
    )
    parser.add_argument("text", nargs="*",
                        help="target caption(s); multiple positional args are joined. "
                             "Skip when using --preset.")
    parser.add_argument("--preset", action="append", default=[],
                        help="apply a genre preset from prompts/genres/<name>.txt; "
                             "each line is trained individually. Repeatable: "
                             "--preset edm --preset techno.")
    parser.add_argument("--weight", type=float, default=5.0,
                        help="per-train weight (default: 5.0)")
    parser.add_argument("--repeat", type=int, default=3,
                        help="train this many times for stronger pull (default: 3)")
    parser.add_argument("--no-allow", action="store_true",
                        help="don't auto-whitelist pop-blocklist terms found in the text")
    args = parser.parse_args(argv)

    paths = Paths.discover()
    if not paths.chain_path.exists():
        print(f"no text chain at {paths.chain_path}; run ./create first", file=sys.stderr)
        return 1

    # Build the list of captions to train: positional text gets joined into
    # one caption, each preset contributes its lines as separate captions.
    captions: list[str] = []
    if args.text:
        joined = " ".join(args.text).strip()
        if joined:
            captions.append(joined)
    for preset in args.preset:
        try:
            captions.extend(_load_preset(preset, paths.prompts_dir))
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if not captions:
        print("nothing to steer with (pass text or --preset NAME)", file=sys.stderr)
        return 1

    chain, order = load_chain(paths.chain_path)
    total_added = 0
    for caption in captions:
        for _ in range(args.repeat):
            total_added += train_text(chain, caption, order, weight=args.weight)
    save_chain(chain, order, paths.chain_path)
    print(
        f"trained text chain: +{total_added} weighted transitions across "
        f"{len(captions)} caption(s) (x{args.repeat} @ weight={args.weight})"
    )
    for caption in captions:
        print(f"  → {caption[:140]}{'...' if len(caption) > 140 else ''}")

    if not args.no_allow:
        full_text = " ".join(captions)
        found = steering_terms_in(full_text)
        if found:
            allowed = load_allowed(paths.state_dir)
            new = found - allowed
            if new:
                allowed |= new
                save_allowed(paths.state_dir, allowed)
                print(f"whitelisted pop terms (now allowed in autofeedback): {sorted(new)}")
            else:
                print(f"pop terms in text already whitelisted: {sorted(found)}")
        else:
            print("no pop-blocklist terms in text — nothing to whitelist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
