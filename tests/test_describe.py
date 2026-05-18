from __future__ import annotations

from markovsound.describe import metadata_caption, strip_training_metadata


def test_metadata_caption_keeps_descriptive_caption_only():
    meta = {
        "caption": "A brittle drone.",
        "bpm": 188,
        "keyscale": "C# major",
        "timesignature": "2",
        "vocal_language": "ar",
    }

    assert metadata_caption(meta) == "A brittle drone."


def test_strip_training_metadata_removes_legacy_suffix():
    old = "A brittle drone, 188 bpm, in C# major, 2 time, ar vocals."

    assert strip_training_metadata(old) == "A brittle drone."
