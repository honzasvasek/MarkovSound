from __future__ import annotations

import json

from markovsound.config import Paths
from markovsound.playback import queued_tracks


def _paths(tmp_path):
    state = tmp_path / "state"
    audio = tmp_path / "Audio"
    return Paths(
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


def test_queued_tracks_excludes_now_playing_item(tmp_path):
    paths = _paths(tmp_path)
    paths.queue_dir.mkdir(parents=True)
    paths.state_dir.mkdir(parents=True)
    for name in ("000001.mp3", "000002.mp3"):
        (paths.queue_dir / name).write_bytes(b"x")
    (paths.state_dir / "now_playing.json").write_text(json.dumps({"file": "000001.mp3"}))

    assert [p.name for p in queued_tracks(paths)] == ["000002.mp3"]
