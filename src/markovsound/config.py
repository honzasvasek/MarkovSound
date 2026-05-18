"""Filesystem layout and ACE server configuration discovery."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    state_dir: Path
    audio_dir: Path
    absorb_dir: Path
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
        state = root / "state"
        audio = root / "Audio"
        prompts = root / "prompts"
        return cls(
            repo_root=root,
            state_dir=state,
            audio_dir=audio,
            absorb_dir=audio / "absorb",
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
