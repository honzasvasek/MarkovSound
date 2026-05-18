"""Filesystem layout, session selection, and ACE server configuration discovery."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


LEGACY_SESSION = "legacy"
SESSION_ENV = "MARKOVSOUND_SESSION"
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    session_name: str
    session_root: Path
    sessions_dir: Path
    current_session_path: Path
    state_dir: Path
    audio_dir: Path
    absorb_dir: Path
    queue_dir: Path
    staging_dir: Path
    archive_dir: Path
    prompts_dir: Path
    chain_path: Path
    cycle_path: Path
    corpus_path: Path
    codes_chain_path: Path
    codes_corpus_path: Path
    lyrics_chain_path: Path
    lyrics_corpus_path: Path
    used_lyrics_corpus_path: Path
    preset_path: Path
    runtime_config_path: Path

    @classmethod
    def discover(cls) -> "Paths":
        root = Path(__file__).resolve().parents[2]
        session_name = _selected_session(root)
        sessions_dir = root / "sessions"
        current_session_path = root / ".markovsound_session"
        session_root = sessions_dir / session_name
        if SESSION_ENV in os.environ:
            # Explicit one-command override: bypass root symlinks.
            state = session_root / "state"
            audio = session_root / "Audio"
        elif (root / "state").is_symlink() or current_session_path.exists():
            # Normal runtime: root state/Audio point at the active session.
            state = root / "state"
            audio = root / "Audio"
        else:
            # Compatibility with old checkouts before sessions existed.
            session_root = root
            state = root / "state"
            audio = root / "Audio"
        prompts = root / "prompts"
        return cls(
            repo_root=root,
            session_name=session_name,
            session_root=session_root,
            sessions_dir=sessions_dir,
            current_session_path=current_session_path,
            state_dir=state,
            audio_dir=audio,
            absorb_dir=audio / "absorb",
            queue_dir=audio / "queue",
            staging_dir=audio / "staging",
            archive_dir=audio / "archive",
            prompts_dir=prompts,
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


def is_valid_session_name(name: str) -> bool:
    return bool(_SESSION_RE.fullmatch(name)) and name not in {".", ".."}


def validate_session_name(name: str) -> str:
    name = name.strip()
    if not is_valid_session_name(name):
        raise ValueError(
            "invalid session name; use letters, numbers, '.', '_' or '-' "
            "and start with a letter or number"
        )
    return name


def _selected_session(root: Path) -> str:
    env_name = os.environ.get(SESSION_ENV, "").strip()
    if env_name:
        return validate_session_name(env_name)
    marker = root / ".markovsound_session"
    try:
        marker_name = marker.read_text(encoding="utf-8").strip()
    except OSError:
        marker_name = ""
    if marker_name:
        return validate_session_name(marker_name)
    legacy_marker = root / "state" / "current_session"
    if not (root / "state").is_symlink():
        try:
            marker_name = legacy_marker.read_text(encoding="utf-8").strip()
        except OSError:
            marker_name = ""
        if marker_name:
            return validate_session_name(marker_name)
    return LEGACY_SESSION


@dataclass(frozen=True)
class AceConfig:
    ace_root: Path
    server_bin: Path
    models_dir: Path
    host: str
    port: int
    lm_model: str
    synth_model: str

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @classmethod
    def discover(cls) -> "AceConfig":
        root = Path(os.environ.get("ACESTEP_ROOT", Path.home() / "Src/acestep.cpp")).expanduser()
        return cls(
            ace_root=root,
            server_bin=root / "build/ace-server",
            models_dir=root / "models",
            host=os.environ.get("ACE_HOST", "127.0.0.1"),
            port=int(os.environ.get("ACE_PORT", "8085")),
            lm_model=os.environ.get("ACE_LM_MODEL", "acestep-5Hz-lm-4B-Q8_0.gguf"),
            synth_model=os.environ.get("ACE_SYNTH_MODEL", "acestep-v15-turbo-Q8_0.gguf"),
        )
