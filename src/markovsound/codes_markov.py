from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from .chain_io import load_chain_payload, save_chain_payload


def parse_codes(codes_str: str) -> list[int]:
    if not codes_str:
        return []
    return [int(x) for x in codes_str.split(",") if x.strip()]


def format_codes(codes: list[int]) -> str:
    return ",".join(str(c) for c in codes)


def build_codes_chain(sequences: list[list[int]], order: int = 1) -> dict:
    chain: dict[tuple[int, ...], dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for seq in sequences:
        if len(seq) <= order:
            continue
        for i in range(order, len(seq)):
            context = tuple(seq[i - order:i])
            chain[context][seq[i]] += 1.0
    return chain


def train_codes(chain: dict, sequence: list[int], order: int = 1, weight: float = 1.0) -> int:
    if len(sequence) <= order:
        return 0
    added = 0
    for i in range(order, len(sequence)):
        context = tuple(sequence[i - order:i])
        chain.setdefault(context, {})
        chain[context][sequence[i]] = chain[context].get(sequence[i], 0.0) + weight
        added += 1
    return added


def untrain_codes(chain: dict, sequence: list[int], order: int = 1) -> int:
    if len(sequence) <= order:
        return 0
    removed = 0
    for i in range(order, len(sequence)):
        context = tuple(sequence[i - order:i])
        if context in chain and sequence[i] in chain[context]:
            chain[context][sequence[i]] -= 1.0
            if chain[context][sequence[i]] <= 0:
                del chain[context][sequence[i]]
            removed += 1
            if not chain[context]:
                del chain[context]
    return removed


def sample_codes(
    chain: dict,
    order: int,
    length: int,
    temperature: float = 1.0,
    max_dead_ends: int = 50,
) -> tuple[list[int], int]:
    """Sample `length` codes from the chain. On a dead-end, jump to a random
    known context — that's the crossover point where one absorbed track's
    trajectory hops onto another's."""
    if not chain:
        raise ValueError("codes chain is empty")
    contexts = list(chain.keys())
    context = list(random.choice(contexts))
    result: list[int] = []
    dead_ends = 0
    while len(result) < length:
        ctx = tuple(context[-order:])
        if ctx not in chain or not chain[ctx]:
            dead_ends += 1
            if dead_ends > max_dead_ends:
                # absurdly stuck; just pad with random codes from any context
                fallback = list(chain.keys())
                while len(result) < length:
                    result.append(random.choice(fallback)[0])
                break
            context = list(random.choice(contexts))
            continue
        choices = chain[ctx]
        codes = list(choices.keys())
        counts = list(choices.values())
        if temperature != 1.0:
            counts = [c ** (1.0 / temperature) for c in counts]
        next_code = random.choices(codes, weights=counts, k=1)[0]
        result.append(next_code)
        context.append(next_code)
    return result, dead_ends


def save_codes_chain(chain: dict, order: int, path: Path) -> None:
    save_chain_payload(chain, order, path)


def load_codes_chain(path: Path) -> tuple[dict, int]:
    return load_chain_payload(path)


def chain_stats(chain: dict) -> dict:
    transitions = sum(len(v) for v in chain.values())
    total_weight = sum(w for v in chain.values() for w in v.values())
    unique_codes = set()
    for ctx, nexts in chain.items():
        unique_codes.update(ctx)
        unique_codes.update(nexts.keys())
    return {
        "contexts": len(chain),
        "transitions": transitions,
        "total_weight": total_weight,
        "unique_codes": len(unique_codes),
    }
