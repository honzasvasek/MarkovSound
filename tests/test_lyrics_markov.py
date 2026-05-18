from __future__ import annotations

from markovsound.lyrics_markov import (
    count_vocal_brackets,
    detokenize_lyrics,
    prune_timestamps,
    tokenize_lyrics,
)


def test_tokenizer_drops_timestamp_brackets_but_keeps_vocal_headers():
    tokens = tokenize_lyrics("[0:00]\n[Verse - Male Vocal]\nla la")

    assert "[0:00]" not in tokens
    assert "[Verse - Male Vocal]" in tokens
    assert count_vocal_brackets(tokens) == 1


def test_detokenize_lyrics_preserves_section_layout():
    tokens = ["[Verse - Male Vocal]", "\n", "la", "la", "!"]

    assert detokenize_lyrics(tokens) == "[Verse - Male Vocal]\nla la!"


def test_prune_timestamps_removes_bad_contexts_and_outputs():
    chain = {
        ("[0:00]",): {"la": 1.0},
        ("ok",): {"[0:01]": 1.0, "la": 1.0},
    }

    stats = prune_timestamps(chain)

    assert stats == {"contexts_removed": 1, "outputs_removed": 1, "transitions_removed": 2}
    assert chain == {("ok",): {"la": 1.0}}
