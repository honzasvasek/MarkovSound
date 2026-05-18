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


def test_publish_output_moves_staged_pair_into_queue(tmp_path):
    from markovsound.config import Paths
    from markovsound.cycle import publish_output

    state = tmp_path / "state"
    audio = tmp_path / "Audio"
    paths = Paths(
        repo_root=tmp_path,
        session_name="legacy",
        session_root=tmp_path,
        sessions_dir=tmp_path / "sessions",
        current_session_path=tmp_path / ".markovsound_session",
        state_dir=state,
        audio_dir=audio,
        absorb_dir=audio / "absorb",
        queue_dir=audio / "queue",
        staging_dir=audio / "staging",
        archive_dir=audio / "archive",
        prompts_dir=tmp_path / "prompts",
        chain_path=state / "markov_chain.pkl",
        cycle_path=state / "cycle.txt",
        corpus_path=state / "prompt_corpus.txt",
        codes_chain_path=state / "codes_chain.pkl",
        codes_corpus_path=state / "codes_corpus.txt",
        lyrics_chain_path=state / "lyrics_chain.pkl",
        lyrics_corpus_path=state / "understood_lyrics.txt",
        used_lyrics_corpus_path=state / "used_lyrics.txt",
        preset_path=state / "preset.txt",
        runtime_config_path=state / "runtime.json",
    )
    paths.staging_dir.mkdir(parents=True)
    staged_mp3 = paths.staging_dir / "000123.mp3"
    staged_json = paths.staging_dir / "000123.json"
    staged_mp3.write_bytes(b"mp3")

    queued_mp3, queued_json = publish_output(
        mp3_path=staged_mp3,
        json_path=staged_json,
        sidecar={"cycle": 123},
        paths=paths,
    )

    assert queued_mp3 == paths.queue_dir / "000123.mp3"
    assert queued_json == paths.queue_dir / "000123.json"
    assert queued_mp3.read_bytes() == b"mp3"
    assert '"cycle": 123' in queued_json.read_text()
    assert not staged_mp3.exists()


def test_transcribe_track_wakes_and_resleeps_external_transcriber(monkeypatch, tmp_path):
    from markovsound import cycle

    events: list[str] = []
    monkeypatch.setattr(cycle.cli_transcribe, "server_alive", lambda _url: True)
    monkeypatch.setattr(cycle.cli_transcribe, "wake", lambda _url: events.append("wake") or True)
    monkeypatch.setattr(cycle.cli_transcribe, "sleep", lambda _url: events.append("sleep") or True)
    monkeypatch.setattr(
        cycle.cli_transcribe,
        "transcribe_one",
        lambda *_args, **_kwargs: {
            "parsed": {"lyrics": "heard", "languages": "nl"},
            "runaway": False,
        },
    )

    result = cycle._transcribe_track(
        mp3_path=tmp_path / "x.mp3",
        sidecar={},
        log=lambda msg: events.append(msg),
    )

    assert result == TranscriptionResult("heard", "nl")
    assert events[0] == "wake"
    assert "transcriber: woke from sleep" in events
    assert events[-2:] == ["sleep", "transcriber: sleeping"]
