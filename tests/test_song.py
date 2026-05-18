from __future__ import annotations

from markovsound.song import _vocalize_caption


def test_vocalize_caption_removes_instrumental_language_and_adds_lead():
    caption = "An instrumental piece with bowed cymbals."

    result = _vocalize_caption(caption, "en")

    assert result.startswith("Song with prominent lead vocals in English over")
    assert "instrumental" not in result.lower()


def test_vocalize_caption_does_not_duplicate_existing_vocal_hint():
    caption = "Raw female vocals over free jazz texture"

    result = _vocalize_caption(caption, "en")

    assert result == "Raw female vocals over free jazz texture."
