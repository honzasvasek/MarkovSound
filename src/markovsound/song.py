"""Song-level prompt shaping and per-take musical scaffold helpers."""
from __future__ import annotations

import random
import re

_LM_CFG_SCALE = 4.0

_LM_NEGATIVE_PROMPT = (
    "cheesy, generic, radio-ready, polished mix, smooth modern production, "
    "verse-chorus-verse, summer vibe, singalong, hook, anthemic"
)

# Used on vocal cycles instead of _LM_NEGATIVE_PROMPT — drops the pop/rock
# negatives (those *describe* vocal music) and pushes against silence only.
# Solo-instrument / free-improv / sound-mass textures stay welcome; we just
# want a singer on top.
_LM_NEGATIVE_PROMPT_VOCAL = (
    "no vocals, no singer, no voice, silent vocal track, instrumental only, "
    "purely instrumental, no lyrics"
)

# Words in the markov caption that contradict a vocal cycle. We only strip
# the literal "instrumental" cluster — "free jazz", "sound mass", "aleatoric"
# are kept because the Zappa-style aesthetic depends on them; vocals just
# need to coexist with that texture.
_INSTRUMENTAL_PHRASES = (
    "an instrumental piece",
    "instrumental piece",
    "purely instrumental",
    "fully instrumental",
    "entirely instrumental",
    "instrumental composition",
    "instrumental track",
    "instrumental",
)

_VOCAL_HINT_WORDS = (
    "vocal", "vocals", "sing", "singer", "singing", "choir", "voice", "voices",
    "chant", "chanting", "rapper", "rapping", "lyrics", "shouting", "humming",
)

# Language → vocal-positive lead phrase. Prepended (not appended) so ACE's
# attention puts it ahead of the rest of the caption — trailing hints get
# overwhelmed by the body of a long markov caption.
_LANG_VOCAL_LEAD = {
    "en": "Song with prominent lead vocals in English",
    "nl": "Nummer met prominente Nederlandse leadzang",
    "fr": "Chanson avec voix principales en français",
    "de": "Lied mit markantem deutschem Leadgesang",
    "es": "Canción con voz principal en español",
    "it": "Canzone con voce principale in italiano",
    "ja": "Song with prominent Japanese-language lead vocals",
}


def _vocalize_caption(caption: str, lang: str) -> str:
    """Rewrite a markov caption to point ACE toward singing.

    Strips 'instrumental'-family phrasing and PREPENDS a language-specific
    "song with lead vocals" lead so the front of the prompt is unambiguous
    about wanting vocals (trailing hints get drowned by long captions).
    """
    rewritten = caption
    for phrase in _INSTRUMENTAL_PHRASES:
        rewritten = re.sub(rf"\b{re.escape(phrase)}\b\.?\s*", "", rewritten, flags=re.IGNORECASE)
    rewritten = re.sub(r"\s+", " ", rewritten).strip(" .,;:-")
    lead = _LANG_VOCAL_LEAD.get(lang.lower(), f"Song with prominent lead vocals in {lang}")
    if not rewritten:
        return lead + "."
    # If the body already declares vocals, don't double up the lead — just
    # ensure we end with a period.
    if any(w in rewritten.lower() for w in _VOCAL_HINT_WORDS):
        return rewritten if rewritten.endswith(".") else rewritten + "."
    # Lowercase the first letter of the body so the concatenation reads as one
    # sentence: "Song with … lead vocals over an explosive jazz piece."
    body = rewritten[0].lower() + rewritten[1:] if rewritten[:1].isupper() else rewritten
    out = f"{lead} over {body}"
    if not out.endswith("."):
        out += "."
    return out

# Wider, more experimental distribution than ACE's own LM would pick.
# Skewed toward slow tempi and odd metres; drone/free-improv friendly.
_BPM_CHOICES = [40, 45, 50, 55, 60, 66, 72, 80, 88, 100, 110, 125, 144, 160, 180, 200, 220]
_KEYSCALE_CHOICES = [
    "C major", "C minor", "C# minor", "D minor", "D# major",
    "E minor", "F major", "F# minor", "G minor", "G# minor",
    "A minor", "B♭ major", "B minor",
]
_TIMESIG_CHOICES = ["3", "4", "5", "7"]


def _experimental_metadata() -> dict[str, object]:
    return {
        "bpm": random.choice(_BPM_CHOICES),
        "keyscale": random.choice(_KEYSCALE_CHOICES),
        "timesignature": random.choice(_TIMESIG_CHOICES),
    }


