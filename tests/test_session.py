from __future__ import annotations

from pathlib import Path

from markovsound import config
from markovsound.cli_session import _activate_session


def _discover_from(root: Path, monkeypatch):
    monkeypatch.setattr(config, "__file__", str(root / "src/markovsound/config.py"))
    return config.Paths.discover()


def test_discover_uses_legacy_layout_before_sessions_exist(tmp_path, monkeypatch):
    paths = _discover_from(tmp_path, monkeypatch)

    assert paths.session_name == "legacy"
    assert paths.session_root == tmp_path
    assert paths.state_dir == tmp_path / "state"
    assert paths.audio_dir == tmp_path / "Audio"


def test_activate_session_moves_legacy_dirs_and_links_active_session(tmp_path, monkeypatch):
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "seed.txt").write_text("x")
    (tmp_path / "Audio").mkdir()
    (tmp_path / "Audio" / "seed.mp3").write_bytes(b"x")
    paths = _discover_from(tmp_path, monkeypatch)

    _activate_session(paths, "live")

    assert (tmp_path / ".markovsound_session").read_text().strip() == "live"
    assert (tmp_path / "state").is_symlink()
    assert (tmp_path / "Audio").is_symlink()
    assert (tmp_path / "sessions/legacy/state/seed.txt").read_text() == "x"
    assert (tmp_path / "sessions/legacy/Audio/seed.mp3").read_bytes() == b"x"
    assert (tmp_path / "state").resolve() == tmp_path / "sessions/live/state"
    assert (tmp_path / "Audio").resolve() == tmp_path / "sessions/live/Audio"
