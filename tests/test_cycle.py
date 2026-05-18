from __future__ import annotations

from markovsound.cycle import SongPlan, TranscriptionResult, _store_take_carryover, lyric_token_bounds


def test_lyric_token_bounds_scale_with_duration():
    assert lyric_token_bounds(30) == (50, 150)
    assert lyric_token_bounds(120) == (80, 240)


def test_store_take_carryover_prefers_transcribed_lyrics():
    state = {"takes_left": 1}

    _store_take_carryover(
        song_state=state,
        meta={"lyrics": "understood", "audio_codes": "1,2"},
        transcription=TranscriptionResult("transcribed", "en"),
    )

    assert state["last_transcribed_lyrics"] == "transcribed"
    assert state["last_audio_codes"] == "1,2"
