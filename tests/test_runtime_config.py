from __future__ import annotations

import json

import pytest

from markovsound.runtime_config import RuntimeConfig, load_runtime_config, save_runtime_config


def test_missing_config_is_created_with_defaults(tmp_path):
    path = tmp_path / "runtime.json"

    cfg = load_runtime_config(path)

    assert cfg == RuntimeConfig()
    assert json.loads(path.read_text())["max_duration_minutes"] == 10


def test_roundtrip_custom_config(tmp_path):
    path = tmp_path / "runtime.json"
    expected = RuntimeConfig(min_duration_minutes=4, max_duration_minutes=6, vocal_prob=0.2)

    save_runtime_config(path, expected)

    assert load_runtime_config(path) == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"min_duration_minutes": 0},
        {"min_duration_minutes": 5, "max_duration_minutes": 4},
        {"caption_min_words": 10, "caption_max_words": 9},
        {"vocal_prob": 1.1},
        {"takes_per_song": 0},
        {"cover_strength": -0.1},
    ],
)
def test_invalid_config_is_rejected(tmp_path, payload):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError):
        load_runtime_config(path)
