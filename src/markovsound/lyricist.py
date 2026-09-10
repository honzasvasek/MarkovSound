"""Lyrics generator backed by `llama-cli` (mainline llama.cpp).

One-shot per call: spawn llama-cli, load model (page-cached after first
use), generate, exit. No persistent server hogging the GPU, so the
generation can share GPU 0 with ace-server (peak ~7 GB DiT + ~5 GB
llama3-8B Q4 fits a 16 GB card).

Llama3's chat template is applied by hand (the --jinja path with -sys
didn't apply the chat template on our system's llama-cli build, so the
model treated -sys as a continuation seed and produced gibberish — raw
prompt with the special tokens works reliably).

Env knobs:
  LLAMA_CLI            : path to llama-cli (default /usr/local/bin/llama-cli)
  LYRICIST_MODEL_PATH  : GGUF path (default: llama3:latest blob in ollama store)
  LYRICIST_CUDA_DEVICE : CUDA device for the lyricist (default 0; share with ace-server)
  LYRICIST_NGL         : layers to offload to GPU (default 99 = all)
  LYRICIST_CTX         : context size (default 4096; raise only if needed)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time

from .config import Paths


LLAMA_CLI = os.environ.get("LLAMA_CLI", "/usr/local/bin/llama-cli")
MODEL_PATH = os.environ.get(
    "LYRICIST_MODEL_PATH",
    "/usr/share/ollama/.ollama/models/blobs/"
    "sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa",
)
CUDA_DEVICE = os.environ.get("LYRICIST_CUDA_DEVICE", "0")
NGL = os.environ.get("LYRICIST_NGL", "99")
CTX = os.environ.get("LYRICIST_CTX", "4096")

# Aimed at Captain Beefheart / Frank Zappa / Sun Ra / Tom Waits / Diamanda
# Galás territory. The earlier "songwriting engineer" prompt produced
# Spotify-grade pop-poetry ("feel the rhythm / take me higher / the city
# at night") that sounded like AI music. This one demands invented syllables,
# non-English mantras, surreal concrete imagery, repetition. Pair with high
# temperature (1.1-1.3) for actual divergence.
SYSTEM_PROMPT = """\
You are an avant-garde lyricist channeling Captain Beefheart, Frank Zappa, Sun Ra, Tom Waits, Yma Sumac, and Diamanda Galás. You write lyrics for the ACE-Step 1.5 music auto-composer. Your job is to make lyrics that are STRANGE — never radio-safe, never AI-pop generic.

### Inputs:
- Caption: genres / instruments / mood
- Length: seconds
- BPM: tempo
- Description: vibe to anchor to

### What to write:
- Mostly INVENTED SYLLABLES and chants — `Sapath thang thang`, `Kulak kumak kumak`, `Ya allah ya allah`, `Wee wee wee`, `Doo bee doo bee doo`, `Aaaaah ah ah`, `Bababa baba`. Coin your own. Mix with broken English fragments.
- Real-or-invented **non-English mantra words**. Don't tell a story.
- **Cryptic spoken-word fragments** as single short sentences: "What is free?", "Losers", "Show your beat!", "One two three four", "The darkness", "Take me to the bus station". Surreal, never autobiographical.
- **Wordless vocal sections**: `(Aaah-ah-ah-ah)`, `(Ooooh-ohhh)`, `(Mmmmm-mmmm)`.
- **Repetition is the engine** — repeat phrases 4–8 times in a row.
- **Concrete weird imagery**: rubber wedge, pickled herring, glass eye, bus station seance, holocaust whisper, sedimenta, harmonica burns, accordion ghost. NOT "love", "freedom", "the night", "the city", "shine bright", "the rhythm".

### Hard rejections — DO NOT produce:
- Standard pop verse-chorus love songs
- "feel the music" / "dance with me" / "take me higher" / "shine bright" / "the rhythm of [anything]" / "raise our hands" / "all night long"
- English narrative storytelling
- Motivational / uplift content
- Polished-songwriter metaphor stacks

### ACE-Step formatting (required):
1. Bracket-tag every section with context-aware vocal cues: `[Verse 1 - Male voice, deep reverb]`, `[Chorus - Layered Female Vocals]`, `[Spoken Word - whispered]`, `[Wordless Vocal Sample - Pitched Down]`, `[Instrumental Break]`, `[Male Vocal Shout]`.
2. Short lines — singable in one breath.
3. Pacing fits Length and BPM (slow = sparser; fast = punchier).

### Output rules — STRICT:
- Output ONLY bracket-tagged sections and their content.
- NO preamble. NEVER start with "Here is", "Here are", "Below is", "Sure,", "Okay,", "I'll write", or any chatter.
- NO postamble. NEVER end with "This draft", "I hope", "Feel free", "Let me know", or any commentary.
- NO quote marks around lines.
- First line of output MUST be a `[` bracket tag.
- Last line of output MUST be either a bracket tag or a lyric line — never prose.\
"""


def load_system_prompt(log=print) -> str:
    """Load the editable lyricist system prompt for the active session."""
    path = Paths.discover().lyrics_prompt_path
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except OSError:
        return SYSTEM_PROMPT
    if not prompt:
        log(f"lyricist: {path} is empty; using built-in prompt")
        return SYSTEM_PROMPT
    return prompt


def _format_llama3_prompt(system: str, user: str) -> str:
    """Llama-3 chat template, hand-rolled. The --jinja flag in our installed
    llama-cli ignores -sys for some reason and just continues the bare prompt
    as raw text — embedding the special tokens here works deterministically.
    """
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _user_message(*, caption: str, duration: float | int,
                  bpm: float | int | None, description: str) -> str:
    parts = [
        f"Caption: {caption}",
        f"Length: {int(round(float(duration)))} seconds",
    ]
    if bpm:
        parts.append(f"BPM: {int(round(float(bpm)))}")
    parts.append("Description: " + (description.strip() or caption))
    return "\n".join(parts)


_END_OF_TEXT_RE = re.compile(r"\[end of text\]|<\|eot_id\|>")

# llama3 routinely ignores "no preamble" in the system prompt — it lands a
# "Here is the optimized lyrics draft..." opener and a "This draft maintains
# a consistent, high-energy pace..." closer. Anchor to the first [bracket]
# line, drop everything before it, then scan past the last [bracket] looking
# for the first paragraph that's clearly prose commentary and drop from
# there.
_PROSE_MARKERS = (
    "this draft", "the lyrics", "feel free", "i hope", "i've structured",
    "this structure", "the structure", "designed to fit", "fits the given",
    "let me know", "hope this", "note:", "this is the", "this version",
    "this should", "this captures", "i've aimed", "aim to evoke",
    "high-energy pace", "structured to fit", "this output", "based on the",
)


def _clean_llama_lyrics(text: str) -> str:
    if not text:
        return text
    lines = text.splitlines()
    bracket_indices = [i for i, l in enumerate(lines) if l.lstrip().startswith("[")]
    if not bracket_indices:
        return text.strip()
    first = bracket_indices[0]
    last = bracket_indices[-1]
    end = len(lines)
    for i in range(last + 1, len(lines)):
        l = lines[i].strip()
        if not l:
            continue
        low = l.lower()
        if any(m in low for m in _PROSE_MARKERS):
            end = i
            break
        # Long single-line prose (>140 chars, sentence-y, no brackets) is
        # almost always commentary too.
        if len(l) > 140 and ". " in l and not l.startswith("["):
            end = i
            break
    cleaned = "\n".join(lines[first:end]).rstrip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def generate_lyrics(
    caption: str,
    duration: float | int,
    bpm: float | int | None = None,
    description: str = "",
    *,
    max_tokens: int = 800,
    temperature: float = 1.2,
    timeout: int = 300,
    log=print,
) -> tuple[str, float]:
    """Spawn llama-cli, generate lyrics, return (text, seconds). Empty text
    on any failure (timeout, non-zero exit, missing binary). Falls through
    silently so the loop never blocks waiting for lyrics."""
    if not os.path.exists(MODEL_PATH):
        log(f"lyricist: model file missing at {MODEL_PATH}; skipping")
        return "", 0.0

    user_msg = _user_message(
        caption=caption, duration=duration, bpm=bpm, description=description,
    )
    prompt = _format_llama3_prompt(load_system_prompt(log=log), user_msg)
    # NOTE: do NOT pass --log-disable here — in this llama-cli build it
    # suppresses the generated stdout too, leaving us with zero bytes.
    # Logs and timing prints land on stderr (which we ignore); generated
    # text stays on stdout.
    cmd = [
        LLAMA_CLI,
        "-m", MODEL_PATH,
        "-p", prompt,
        "-n", str(max_tokens),
        "-no-cnv",
        "--no-display-prompt",
        "--no-warmup",
        "-r", "<|eot_id|>",
        "-ngl", NGL,
        "--temp", str(temperature),
        "-c", CTX,
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": CUDA_DEVICE}
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, env=env, timeout=timeout,
        )
    except FileNotFoundError:
        log(f"lyricist: {LLAMA_CLI} not found on PATH")
        return "", 0.0
    except subprocess.TimeoutExpired:
        log(f"lyricist: timed out after {timeout}s")
        return "", round(time.time() - t0, 2)
    elapsed = round(time.time() - t0, 2)
    if proc.returncode != 0:
        log(f"lyricist: llama-cli exited {proc.returncode}; "
            f"stderr tail: {proc.stderr[-200:].decode('utf-8', errors='replace')!r}")
        return "", elapsed
    text = proc.stdout.decode("utf-8", errors="replace")
    text = _END_OF_TEXT_RE.sub("", text)
    text = _clean_llama_lyrics(text)
    # Strip pure-timecode brackets and stop-the-music markers (same as
    # cli_transcribe does for transcriber output). The lyricist often
    # invents fake timestamps like [2:90-300] that aren't real; ACE just
    # reads them as instrumental pauses.
    try:
        from . import cli_transcribe
        text = cli_transcribe.strip_silence_markers(
            cli_transcribe.strip_timecodes(text)
        )
    except Exception:  # noqa: BLE001 — never let cleanup kill the loop
        pass
    return text, elapsed


def main(argv: list[str] | None = None) -> int:
    """Standalone CLI: ./lyricist "caption" [-d 120] [-b 124] [--description …]"""
    import argparse
    parser = argparse.ArgumentParser(prog="lyricist",
                                     description="Generate ACE-Step lyrics via llama-cli.")
    parser.add_argument("caption", help="genre/instruments/mood caption")
    parser.add_argument("-d", "--duration", type=float, default=180.0,
                        help="target duration in seconds (default 180)")
    parser.add_argument("-b", "--bpm", type=float, default=None)
    parser.add_argument("--description", default="",
                        help="optional draft or vibe description (defaults to caption)")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args(argv)
    lyrics, secs = generate_lyrics(
        args.caption, args.duration, args.bpm, args.description,
        max_tokens=args.max_tokens, temperature=args.temperature,
    )
    print(f"=== lyricist (llama-cli, {secs}s) ===\n{lyrics}")
    return 0 if lyrics else 1


if __name__ == "__main__":
    sys.exit(main())
