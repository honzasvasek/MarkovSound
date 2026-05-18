from __future__ import annotations

import pickle
import random
import re
from collections import defaultdict
from pathlib import Path


START = "__START__"
END = "__END__"
MIN_WEIGHT = 1e-4


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+|[,;:.!?]", text.lower())


def build_chain(captions: list[str], order: int) -> dict:
    chain: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for caption in captions:
        words = tokenize(caption)
        if not words:
            continue
        padded = [START] * order + words + [END]
        for index in range(order, len(padded)):
            context = tuple(padded[index - order:index])
            chain[context][padded[index]] += 1
    return chain


def cleanup_chain(chain: dict) -> None:
    empty_states = []
    for state, nexts in chain.items():
        dead = [word for word, weight in nexts.items() if weight < MIN_WEIGHT]
        for word in dead:
            del nexts[word]
        if not nexts:
            empty_states.append(state)
    for state in empty_states:
        del chain[state]


def sample(choices: dict[str, float], temperature: float) -> str:
    if not choices:
        return END
    words = list(choices.keys())
    counts = list(choices.values())
    if temperature == 1.0:
        return random.choices(words, weights=counts, k=1)[0]
    weights = [count ** (1.0 / temperature) for count in counts]
    return random.choices(words, weights=weights, k=1)[0]


def detokenize(words: list[str]) -> str:
    result: list[str] = []
    for word in words:
        if word in {",", ";", ":", ".", "!", "?"} and result:
            result[-1] = result[-1] + word
        else:
            result.append(word)
    if result:
        result[0] = result[0].capitalize()
    text = " ".join(result)
    if text and text[-1] not in ".!?":
        last = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last > 0:
            text = text[: last + 1]
    return text


def generate_caption(
    chain: dict,
    order: int,
    min_words: int = 12,
    max_words: int = 80,
    temperature: float = 1.0,
    max_attempts: int = 50,
) -> str | None:
    for _ in range(max_attempts):
        context = [START] * order
        words: list[str] = []
        for _ in range(max_words):
            ctx = tuple(context)
            if ctx not in chain:
                context = [START] * order
                continue
            next_word = sample(chain[ctx], temperature)
            if next_word == END:
                if len(words) >= min_words:
                    break
                context = [START] * order
                continue
            words.append(next_word)
            context = context[1:] + [next_word]
        if len(words) >= min_words:
            return detokenize(words)
    return None


def train_text(chain: dict, text: str, order: int, weight: float = 1.0) -> int:
    words = tokenize(text)
    if not words:
        return 0
    padded = [START] * order + words + [END]
    updated = 0
    for index in range(order, len(padded)):
        context = tuple(padded[index - order:index])
        next_word = padded[index]
        chain.setdefault(context, {})
        chain[context][next_word] = chain[context].get(next_word, 0.0) + weight
        updated += 1
    return updated


def untrain_text(chain: dict, text: str, order: int) -> int:
    words = tokenize(text)
    if not words:
        return 0
    padded = [START] * order + words + [END]
    updated = 0
    for index in range(order, len(padded)):
        context = tuple(padded[index - order:index])
        next_word = padded[index]
        if context in chain and next_word in chain[context]:
            chain[context][next_word] -= 1
            updated += 1
    cleanup_chain(chain)
    return updated


def save_chain(chain: dict, order: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"chain": dict(chain), "order": order}, handle)


def load_chain(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    return data["chain"], data["order"]
