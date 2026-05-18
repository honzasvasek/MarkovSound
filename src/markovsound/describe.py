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



_TRAINING_METADATA_SUFFIX_RE = re.compile(
    r",\s*\d+\s+bpm,\s*in\s+[^,]+,\s*\d+\s*time"
    r"(?:,\s*[a-z]{2}\s+vocals)?\.?$",
    re.IGNORECASE,
)


def strip_training_metadata(text: str) -> str:
    """Remove legacy appended BPM/key/time/vocal metadata from a caption."""
    cleaned = _TRAINING_METADATA_SUFFIX_RE.sub("", text.strip()).rstrip(" ,")
    if cleaned and not cleaned.endswith("."):
        cleaned += "."
    return cleaned

def metadata_caption(meta: dict[str, Any]) -> str:
    """Return only the descriptive /understand caption for text-chain training.

    Older versions appended BPM/key/time/language metadata here. In a Markov
    text chain those suffixes become reusable fragments (`bpm, in d minor,
    time, no vocals`) rather than useful musical guidance, so metadata now
    stays in sidecars and the caption chain learns prose only.
    """
    caption = (meta.get("caption") or "").strip()
    return strip_training_metadata(caption)
