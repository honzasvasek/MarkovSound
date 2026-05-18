"""Seed the lyrics Markov chain with synthetic Zappa-style passages that
contain dense Male/Female/Choir/Spoken vocal headers.

Run once to bias the chain toward generating more vocal sections. Each
synthetic passage mixes a couple of instrumental stage directions with
vocal-headed verses + placeholder syllables, so the chain learns the
pattern "stage direction → vocal header → sung syllables" in many
variations.
"""
from __future__ import annotations

import random
import sys
from .config import Paths
from .lyrics_markov import (
    load_lyrics_chain,
    save_lyrics_chain,
    train_lyrics,
    count_vocal_brackets,
    tokenize_lyrics,
)


_MALE_HEADERS = [
    "[Verse 1 - Male Vocal]",
    "[Verse 2 - Male Vocal]",
    "[Verse 3 - Male Vocal]",
    "[Chorus - Male Vocal]",
    "[Pre-Chorus - Male Vocal]",
    "[Bridge - Male Vocal]",
    "[Outro - Male Vocal Harmony]",
    "[Intro - Male Vocal]",
    "[Male Vocal Shout]",
    "[Male Vocal Cry]",
    "[Male Vocal Whisper]",
    "[Spoken Word - Male Voice]",
    "[Male Vocal Sample - pitched up]",
    "[Male Vocal Sample - processed]",
    "[Male Lead Vocal Soaring]",
]

_FEMALE_HEADERS = [
    "[Verse 1 - Female Vocal]",
    "[Verse 2 - Female Vocal]",
    "[Verse 3 - Female Vocal]",
    "[Chorus - Female Vocal]",
    "[Pre-Chorus - Female Vocal]",
    "[Bridge - Female Vocal]",
    "[Outro - Female Vocal]",
    "[Intro - Female Vocal]",
    "[Female Vocal Shout]",
    "[Female Vocal Cry]",
    "[Female Vocal Whisper]",
    "[Spoken Word - Female Voice]",
    "[Female Vocal Sample - pitched down]",
    "[Female Vocal Sample - processed]",
    "[Female Lead Vocal Soaring]",
    "[Female Operatic Vocal]",
]

_CHOIR_HEADERS = [
    "[Choir - Layered Harmonies]",
    "[Choir Vocal Build-up]",
    "[Choir - Wordless Vowels]",
    "[Verse - Mixed Choir]",
    "[Chorus - Full Choir]",
    "[Backing Vocals - Female Choir]",
    "[Backing Vocals - Male Choir]",
]

_SPOKEN_HEADERS = [
    "[Spoken Word]",
    "[Spoken Word Sample]",
    "[Spoken word, filtered voice]",
    "[Spoken word, whispered]",
    "[Spoken Word - Reverberant]",
    "[Spoken Word Intro]",
]

_RAP_HEADERS = [
    "[Verse - Rap]",
    "[Chorus - Rap Vocal]",
    "[Rap Verse - Male]",
    "[Rap Verse - Female]",
]

# Placeholder sung content — short, flexible syllables ACE happily renders.
_SUNG_LINES = [
    ["la la la la la", "oh oh oh oh", "la la la la la la"],
    ["doo doo doo doo", "doo wah doo wah", "doo doo doo"],
    ["hey ya hey ya", "hey ya hey ya hey", "hey ya ya"],
    ["oooh ahhh oooh", "ahhh oooh ahhh", "oooh oooh oooh"],
    ["yeah yeah yeah", "yeah yeah", "oh yeah oh yeah"],
    ["na na na na na", "na na na na", "na na na"],
    ["sha la la la", "sha la la la la", "sha la la"],
    ["bababa baba", "bababa baba ba", "baba baba"],
    ["ooh la la", "ooh la la la", "la la ooh la"],
    ["mmm mmm mmm", "mmm aaah mmm", "mmm hmm hmm"],
    ["aaah aaah aaah", "aaah ah ah", "aaah ooh aaah"],
    ["one two three four", "one two three", "four three two one"],
    ["go go go", "let's go", "go on go on"],
    ["come on come on", "come on now", "come on come on come on"],
    ["hey hey hey", "ho hey ho hey", "hey ho hey ho"],
    ["wo oh wo oh", "wo oh wo oh wo", "wo oh"],
    ["ay ay ay", "ay ay ay ay", "ay ay"],
    ["ya allah ya allah", "ya allah", "ya allah ya allah ya allah"],
    ["sing it loud", "sing it loud sing it loud", "sing"],
    ["take me higher", "take me higher higher", "take me up"],
    ["dance with me", "dance dance dance", "dance with me tonight"],
    ["forever and ever", "forever", "forever and ever amen"],
]

# A few descriptive stage directions to interleave so chains stay Zappa-ish.
_STAGE_DIRECTIONS = [
    "[Instrumental Section]",
    "[Music: Saxophone solo continues]",
    "[Instrumental Break]",
    "[Bridge]",
    "[Intro]",
    "[Outro]",
    "[Drum fill]",
    "[Layered synth pads enter]",
    "[Full band drops out]",
    "[Build-up with layered guitars]",
    "[Beat drop]",
]


def _make_passage(rng: random.Random) -> str:
    """One synthetic Zappa-style passage with 2-4 vocal sections."""
    n_vocal = rng.randint(2, 4)
    parts: list[str] = []

    # 50% chance of opening with a stage direction.
    if rng.random() < 0.5:
        parts.append(rng.choice(_STAGE_DIRECTIONS))

    for _ in range(n_vocal):
        category = rng.choices(
            ["male", "female", "choir", "spoken", "rap"],
            weights=[35, 35, 12, 12, 6],
            k=1,
        )[0]
        if category == "male":
            header = rng.choice(_MALE_HEADERS)
        elif category == "female":
            header = rng.choice(_FEMALE_HEADERS)
        elif category == "choir":
            header = rng.choice(_CHOIR_HEADERS)
        elif category == "spoken":
            header = rng.choice(_SPOKEN_HEADERS)
        else:
            header = rng.choice(_RAP_HEADERS)

        parts.append(header)
        sung = rng.choice(_SUNG_LINES)
        parts.extend(sung)

        # Sometimes a brief stage direction between vocal sections.
        if rng.random() < 0.4:
            parts.append("")  # blank line
            parts.append(rng.choice(_STAGE_DIRECTIONS))

    if rng.random() < 0.5:
        parts.append("")
        parts.append(rng.choice(_STAGE_DIRECTIONS))

    return "\n".join(parts)


def main() -> int:
    paths = Paths.discover()
    if paths.lyrics_chain_path.exists():
        chain, order = load_lyrics_chain(paths.lyrics_chain_path)
        print(f"loaded chain: {len(chain)} contexts, order={order}")
    else:
        chain, order = {}, 2
        print("no chain yet; creating order=2")

    rng = random.Random(42)
    n_passages = 200
    total_added = 0
    total_vocal_brackets = 0
    for _ in range(n_passages):
        passage = _make_passage(rng)
        total_vocal_brackets += count_vocal_brackets(tokenize_lyrics(passage))
        # weight=1.5 so synthetic passages have a bit more pull than a single
        # real /lm or /understand absorption (weight=1.0).
        total_added += train_lyrics(chain, passage, order, weight=1.5)

    save_lyrics_chain(chain, order, paths.lyrics_chain_path)
    print(
        f"trained {n_passages} synthetic passages "
        f"({total_vocal_brackets} vocal headers, "
        f"+{total_added} weighted transitions)"
    )
    print(f"chain now: {len(chain)} contexts, "
          f"{sum(len(v) for v in chain.values())} transitions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
