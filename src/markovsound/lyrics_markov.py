"""Markov chain over ACE-Step lyrics.

Lyrics have structure that captions don't: bracketed section headers / stage
directions, line breaks, sometimes empty lines. The tokenizer treats each
`[…]` block and each newline as a single token alongside words and punctuation,
so the chain learns when to break lines and switch sections.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path

from .chain_io import load_chain_payload, save_chain_payload


START = "__START__"
END = "__END__"

_TOKEN_RE = re.compile(r"\[[^\]]*\]|[A-Za-z']+|[.,;:!?]")

# A bracket counts as a *vocal section header* if it mentions any of these:
# [Verse 2 - Male Vocal], [Female vocal], [Choir], [Spoken word sample: ...],
# [Male Vocal Shout], etc. The Zappa-style descriptive brackets stay — we just
# need at least one of THESE so ACE actually sings somewhere in the piece.
_VOCAL_BRACKET_RE = re.compile(
    r"\b(vocal|vocals|voice|voices|choir|chorus\s+vocal|spoken|rap|"
    r"chant|sing|sung|sings|singer|singing|hum|humming|whisper|whispers|"
    r"whispered|shout|shouts|shouted|scream|screams|screamed)\b",
    re.IGNORECASE,
)

# Pure-timestamp brackets — chain-noise that contributes nothing to ACE:
#   [2:58]   [0:00]   [8:31-8:47]   [0:00 - 0:30]   [1:07 - 1:31]
# Mixed brackets like "[0:31-0:47 - Build-up]" still have description and
# are intentionally NOT matched.
_TIMESTAMP_BRACKET_RE = re.compile(
    r"^\[\s*\d+:\d+(?:\s*-\s*\d+:\d+)?\s*\]$"
)


def is_timestamp_bracket(token: str) -> bool:
    return bool(_TIMESTAMP_BRACKET_RE.match(token))


def tokenize_lyrics(text: str) -> list[str]:
    tokens: list[str] = []
    lines = text.split("\n")
    for index, line in enumerate(lines):
        for match in _TOKEN_RE.findall(line):
            # Drop pure-timestamp brackets at the tokenizer — they're chain
            # noise that ACE ignores and that bloats both the chain and the
            # in-player lyrics view.
            if is_timestamp_bracket(match):
                continue
            tokens.append(match)
        if index < len(lines) - 1:
            tokens.append("\n")
    return tokens


_PUNCT = {",", ";", ":", ".", "!", "?"}


def _is_word_token(token: str) -> bool:
    return bool(token) and token[0].isalpha() and not token.startswith("[")


def _is_bracket_token(token: str) -> bool:
    return token.startswith("[") and token.endswith("]")


def is_vocal_bracket(token: str) -> bool:
    """True for stage-direction brackets that mark a sung/spoken section.

    The Zappa-style descriptive brackets ([Instrumental Section 4],
    [Music starts], etc.) are KEPT — we only require that at least one
    of THESE vocal-section markers also appears, so ACE actually opens
    its mouth somewhere in the piece.
    """
    if not _is_bracket_token(token):
        return False
    return bool(_VOCAL_BRACKET_RE.search(token))


def count_word_tokens(tokens) -> int:
    if isinstance(tokens, str):
        tokens = tokenize_lyrics(tokens)
    return sum(1 for t in tokens if _is_word_token(t))


def count_vocal_brackets(tokens) -> int:
    if isinstance(tokens, str):
        tokens = tokenize_lyrics(tokens)
    return sum(1 for t in tokens if is_vocal_bracket(t))


def detokenize_lyrics(tokens: list[str]) -> str:
    out: list[str] = []
    line_started = False
    for token in tokens:
        if token == "\n":
            out.append("\n")
            line_started = False
        elif token in _PUNCT and line_started:
            out[-1] = out[-1] + token
        elif token.startswith("[") and token.endswith("]"):
            if line_started:
                out.append("\n")
            out.append(token)
            line_started = True
        else:
            if line_started and (not out or not out[-1].endswith("\n")):
                out.append(" " + token)
            else:
                out.append(token)
            line_started = True
    return "".join(out).strip()


def build_chain(corpus: list[str], order: int) -> dict:
    chain: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for lyric in corpus:
        tokens = tokenize_lyrics(lyric)
        if not tokens:
            continue
        padded = [START] * order + tokens + [END]
        for i in range(order, len(padded)):
            ctx = tuple(padded[i - order:i])
            chain[ctx][padded[i]] += 1.0
    return chain


def train_lyrics(
    chain: dict,
    text: str,
    order: int,
    weight: float = 1.0,
    vocal_bracket_boost: float = 2.5,
) -> int:
    """Train chain on `text`. Transitions that LAND ON a vocal-section
    bracket ([Verse 2 - Male Vocal] etc.) or land on a word right after
    one are weighted higher, so the chain remembers vocal openings without
    losing the Zappa-style stage directions around them.
    """
    tokens = tokenize_lyrics(text)
    if not tokens:
        return 0
    padded = [START] * order + tokens + [END]
    added = 0
    for i in range(order, len(padded)):
        ctx = tuple(padded[i - order:i])
        tgt = padded[i]
        w = weight
        if is_vocal_bracket(tgt):
            w *= vocal_bracket_boost
        elif _is_word_token(tgt) and any(is_vocal_bracket(c) for c in ctx):
            # First few words right after a vocal header → the actual sung
            # line. Boost so the chain learns to follow vocal openings
            # with real syllables instead of veering into more brackets.
            w *= vocal_bracket_boost
        chain.setdefault(ctx, {})
        chain[ctx][tgt] = chain[ctx].get(tgt, 0.0) + w
        added += 1
    return added


def sample_lyrics(
    chain: dict,
    order: int,
    min_tokens: int = 60,
    max_tokens: int = 400,
    temperature: float = 1.0,
    max_attempts: int = 30,
    min_vocal_sections: int = 2,
    min_sung_words: int = 12,
    vocal_bracket_boost: float = 2.0,
    bracket_repeat_cap: int = 5,
) -> str | None:
    """Sample lyrics from the chain, preserving Zappa-style stage directions.

    During sampling, transitions whose target is a vocal-section bracket
    ([Verse 2 - Male Vocal] etc.) get their probability multiplied by
    `vocal_bracket_boost`. This pushes the sampler to insert vocal openings
    more often without losing the surrounding instrumental texture.

    To stop the order-2 chain from getting stuck in tight cycles
    (`[Spoken Word]\\n[Male vocal, layered]\\n[Music starts]\\n` × 30 etc.),
    any bracket token already emitted `bracket_repeat_cap` times this sample
    is masked from subsequent draws — surrounding word repeats ("la la la")
    are untouched.

    Accepts only samples with ≥`min_vocal_sections` vocal headers and
    ≥`min_sung_words` word tokens.
    """
    if not chain:
        return None
    best: tuple[tuple[int, int], list[str]] | None = None
    for _ in range(max_attempts):
        ctx = [START] * order
        tokens: list[str] = []
        bracket_counts: dict[str, int] = {}
        consecutive_resets = 0
        while len(tokens) < max_tokens:
            key = tuple(ctx)
            if key not in chain or not chain[key]:
                consecutive_resets += 1
                if consecutive_resets > 8:
                    break
                ctx = [START] * order
                continue
            choices = chain[key]
            cand = list(choices.keys())
            counts_raw = list(choices.values())
            counts = []
            for w, c in zip(cand, counts_raw):
                weight = c ** (1.0 / temperature) if temperature != 1.0 else c
                if _is_bracket_token(w) and bracket_counts.get(w, 0) >= bracket_repeat_cap:
                    weight = 0.0
                elif is_vocal_bracket(w):
                    weight *= vocal_bracket_boost
                counts.append(weight)
            if sum(counts) <= 0:
                # All candidates capped — try a restart, but cap consecutive
                # restarts to avoid spinning if START → caps too.
                consecutive_resets += 1
                if consecutive_resets > 8:
                    break
                ctx = [START] * order
                continue
            consecutive_resets = 0
            nxt = random.choices(cand, weights=counts, k=1)[0]
            if nxt == END:
                if len(tokens) >= min_tokens:
                    break
                consecutive_resets += 1
                if consecutive_resets > 8:
                    break
                ctx = [START] * order
                continue
            tokens.append(nxt)
            if _is_bracket_token(nxt):
                bracket_counts[nxt] = bracket_counts.get(nxt, 0) + 1
            ctx = ctx[1:] + [nxt]
        if len(tokens) < min_tokens:
            continue
        vocal_sections = count_vocal_brackets(tokens)
        sung_words = count_word_tokens(tokens)
        score = (vocal_sections, sung_words)
        if not best or score > best[0]:
            best = (score, tokens)
        if vocal_sections >= min_vocal_sections and sung_words >= min_sung_words:
            return detokenize_lyrics(tokens)
    if best and best[0][0] >= min_vocal_sections and best[0][1] >= min_sung_words:
        return detokenize_lyrics(best[1])
    return None


def untrain_lyrics(chain: dict, text: str, order: int) -> int:
    tokens = tokenize_lyrics(text)
    if not tokens:
        return 0
    padded = [START] * order + tokens + [END]
    removed = 0
    for i in range(order, len(padded)):
        ctx = tuple(padded[i - order:i])
        if ctx in chain and padded[i] in chain[ctx]:
            chain[ctx][padded[i]] -= 1.0
            if chain[ctx][padded[i]] <= 0:
                del chain[ctx][padded[i]]
            removed += 1
            if not chain[ctx]:
                del chain[ctx]
    return removed

def save_lyrics_chain(chain: dict, order: int, path: Path) -> None:
    save_chain_payload(chain, order, path)


def load_lyrics_chain(path: Path) -> tuple[dict, int]:
    return load_chain_payload(path)


def chain_stats(chain: dict) -> dict:
    transitions = sum(len(v) for v in chain.values())
    return {"contexts": len(chain), "transitions": transitions}


def prune_timestamps(chain: dict) -> dict:
    """Strip pure-timestamp bracket tokens out of an existing chain.

    Returns counters {"contexts_removed", "transitions_removed",
    "outputs_removed"}. Mutates `chain` in place.

    Strategy:
      - Remove any context where the order-N history contains a timestamp
        token (those contexts only exist *because* the chain went through
        a timestamp once and can't be reached cleanly anyway).
      - For surviving contexts, drop any output transition whose target is
        a timestamp token.
      - Drop contexts that end up with no outputs.
    """
    contexts_removed = 0
    outputs_removed = 0
    transitions_removed = 0

    for ctx in list(chain.keys()):
        if any(is_timestamp_bracket(tok) for tok in ctx):
            n_outs = len(chain[ctx])
            contexts_removed += 1
            transitions_removed += n_outs
            del chain[ctx]
            continue
        outs = chain[ctx]
        bad = [t for t in outs if is_timestamp_bracket(t)]
        for t in bad:
            del outs[t]
            outputs_removed += 1
            transitions_removed += 1
        if not outs:
            contexts_removed += 1
            del chain[ctx]

    return {
        "contexts_removed": contexts_removed,
        "outputs_removed": outputs_removed,
        "transitions_removed": transitions_removed,
    }




def replace_string(chain: dict, old: str, new: str) -> dict[str, int]:
    """Replace a string inside every lyrics-chain token.

    Lyrics-chain tokens are words, punctuation, newlines, or whole bracket
    headers. This means replacements can rewrite a whole token (`la`→`na`) or
    part of a bracket header (`Rap`→`Spoken Rap`). Contexts and outgoing
    transitions can collide after replacement; when they do, their weights are
    summed so no learned probability mass is lost.
    """
    rebuilt: dict[tuple, dict[str, float]] = {}
    contexts_changed = 0
    outputs_changed = 0
    merged_contexts = 0
    merged_outputs = 0

    for context, nexts in chain.items():
        replaced_context = tuple(token.replace(old, new) for token in context)
        if replaced_context != context:
            contexts_changed += 1
        if replaced_context in rebuilt:
            merged_contexts += 1
        target_nexts = rebuilt.setdefault(replaced_context, {})
        for token, weight in nexts.items():
            replaced_token = token.replace(old, new)
            if replaced_token != token:
                outputs_changed += 1
            if replaced_token in target_nexts:
                merged_outputs += 1
            target_nexts[replaced_token] = target_nexts.get(replaced_token, 0.0) + weight

    chain.clear()
    chain.update(rebuilt)
    return {
        "contexts_changed": contexts_changed,
        "outputs_changed": outputs_changed,
        "merged_contexts": merged_contexts,
        "merged_outputs": merged_outputs,
    }
