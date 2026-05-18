"""Caption extraction and pop-style rejection heuristics for feedback training."""
from __future__ import annotations

import re
from typing import Any


METADATA_FIELDS = ("bpm", "keyscale", "timesignature", "vocal_language", "duration")

# When /understand describes the rendered audio in these terms, ACE-Step
# probably regressed toward its commercial-music defaults. Training those
# words back into the chain would slowly pull the seed vocabulary in the
# same direction. We skip the training step when too many appear at once.
_POP_WORDS = frozenset({
    "catchy", "anthemic", "chorus", "hook", "danceable", "singalong",
    "drop", "festival", "stadium", "club", "banger", "groove",
    "verse", "pop", "rock", "edm", "trap", "rap", "ska", "country",
    "blues", "funk", "soul", "gospel", "disco", "house", "techno",
    "trance", "dubstep", "reggae", "dub", "vaporwave", "chillwave",
    "afrobeats", "gqom", "synthpop", "supersaw", "arpeggiated",
    "polished", "summer", "upbeat", "energetic", "infectious",
    "irresistible", "wailing", "harmonica",
})
# Multi-word phrases the single-word tokenizer can't catch.
_POP_PHRASES = (
    "hip hop", "hip-hop", "k-pop", "j-pop", "lo-fi", "lo fi", "lofi",
    "boom bap", "boom-bap", "four on the floor", "four-on-the-floor",
    "high energy", "high-energy", "feel good", "feel-good",
    "radio ready", "radio-ready", "hard trance", "hard rock",
    "verse chorus", "verse-chorus", "bass drop", "synth lead",
    "kick drum", "hi-hats", "hi hats", "808",
)
_POP_THRESHOLD = 2


def _pop_score(text: str, allowed: set[str] | None = None) -> int:
    allowed = allowed or set()
    lower = text.lower()
    tokens = set(re.findall(r"[a-zA-Z]+", lower))
    word_hits = sum(1 for t in tokens if t in _POP_WORDS and t not in allowed)
    phrase_hits = sum(1 for p in _POP_PHRASES if p in lower and p not in allowed)
    return word_hits + phrase_hits


def is_too_pop(text: str, allowed: set[str] | None = None) -> bool:
    return _pop_score(text, allowed) >= _POP_THRESHOLD


def steering_terms_in(text: str) -> set[str]:
    """All pop-blocklist words/phrases present in `text`. Used by ./steer to
    auto-whitelist whatever the user explicitly asked for."""
    lower = text.lower()
    tokens = set(re.findall(r"[a-zA-Z]+", lower))
    hits = {t for t in tokens if t in _POP_WORDS}
    hits.update(p for p in _POP_PHRASES if p in lower)
    return hits


def metadata_caption(meta: dict[str, Any]) -> str:
    """Turn /understand output into a training caption.

    We feed the *caption* and selected metadata fields back into the chain;
    lyrics are intentionally ignored (they belong to a separate corpus).
    """
    parts: list[str] = []
    caption = (meta.get("caption") or "").strip()
    if caption:
        parts.append(caption.rstrip("."))
    extras: list[str] = []
    bpm = meta.get("bpm")
    if isinstance(bpm, (int, float)) and bpm > 0:
        extras.append(f"{int(round(bpm))} bpm")
    key = (meta.get("keyscale") or "").strip()
    if key:
        extras.append(f"in {key}")
    ts = (meta.get("timesignature") or "").strip()
    if ts:
        extras.append(f"{ts} time")
    lang = (meta.get("vocal_language") or "").strip().lower()
    # zxx = no linguistic content; und/unknown/mul = LM couldn't tell.
    # Drop anything that isn't a plausible 2-letter ISO 639-1 code.
    if len(lang) == 2 and lang.isalpha() and lang not in {"zz"}:
        extras.append(f"{lang} vocals")
    if extras:
        parts.append(", ".join(extras))
    if not parts:
        return ""
    text = ", ".join(parts).strip()
    if not text.endswith("."):
        text += "."
    return text
