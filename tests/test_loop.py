from __future__ import annotations

from markovsound.loop import log


def test_loop_logger_is_available(capsys):
    log("hello")

    assert capsys.readouterr().out.endswith("hello\n")


def test_reload_changed_lyrics_chain_picks_up_external_edit(tmp_path):
    from markovsound.config import Paths
    from markovsound.loop import ChainWatch, _reload_changed_chains
    from markovsound.lyrics_markov import save_lyrics_chain

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
    save_lyrics_chain({("old",): {"old": 1.0}}, 1, paths.lyrics_chain_path)
    watches = {
        "text": ChainWatch.from_path(paths.chain_path),
        "codes": ChainWatch.from_path(paths.codes_chain_path),
        "lyrics": ChainWatch.from_path(paths.lyrics_chain_path),
    }
    # Simulate replace-lyrics writing a new chain file while create is alive.
    save_lyrics_chain({("new",): {"new": 1.0}}, 1, paths.lyrics_chain_path)

    _chain, _order, _codes, _codes_order, lyrics, lyrics_order = _reload_changed_chains(
        paths=paths,
        chain={},
        order=1,
        codes_chain={},
        codes_order=1,
        lyrics_chain={("old",): {"old": 1.0}},
        lyrics_order=1,
        watches=watches,
        log=lambda _msg: None,
    )

    assert lyrics == {("new",): {"new": 1.0}}
    assert lyrics_order == 1
