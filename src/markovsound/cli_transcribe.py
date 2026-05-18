"""Transcribe audio with ACE-Step transcriber served by vLLM.

Posts mp3/wav files as base64 to a vLLM OpenAI-compatible /v1/chat/completions
endpoint, using the prompt the model card specifies:

    *Task* Transcribe this audio in detail

Saves the raw text response and a parsed view to JSONL.

Server setup (run separately, needs ~13 GB VRAM):

    pip install vllm
    vllm serve Civitai/acestep-transcriber-FP8 \\
        --host 0.0.0.0 --port 8000 \\
        --max-model-len 32768 --gpu-memory-utilization 0.9 \\
        --trust-remote-code

Endpoint can be overridden via TRANSCRIBER_URL (default http://127.0.0.1:8000)
and TRANSCRIBER_MODEL (default Civitai/acestep-transcriber-FP8).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

from .config import Paths


DEFAULT_BASE_URL = os.environ.get("TRANSCRIBER_URL", "http://127.0.0.1:8001")
DEFAULT_MODEL = os.environ.get("TRANSCRIBER_MODEL", "Civitai/acestep-transcriber-FP8")
PROMPT = "*Task* Transcribe this audio in detail"
# vLLM's audio-input shape mirrors the OpenAI audio-in beta: each user message
# carries an `input_audio` part with base64 data + a format tag. Some vLLM
# builds use `audio_url` instead; if your server rejects this shape, set
# TRANSCRIBER_AUDIO_FIELD=audio_url.
AUDIO_FIELD = os.environ.get("TRANSCRIBER_AUDIO_FIELD", "input_audio")


def server_alive(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=3)
        return r.ok
    except requests.RequestException:
        return False


def _build_audio_part(audio_b64: str, fmt: str) -> dict:
    if AUDIO_FIELD == "audio_url":
        return {
            "type": "audio_url",
            "audio_url": {"url": f"data:audio/{fmt};base64,{audio_b64}"},
        }
    # OpenAI-style "input_audio" (the vLLM default for Qwen2.5-Omni)
    return {
        "type": "input_audio",
        "input_audio": {"data": audio_b64, "format": fmt},
    }


def _to_wav_bytes(audio_path: Path) -> bytes:
    """Decode any audio file to 16 kHz mono 16-bit PCM WAV via ffmpeg.

    vLLM's audio loader tries soundfile first then pyav. soundfile handles
    WAV natively without needing the (often-missing) mp3 codecs, so by the
    time the request lands at the server it just works regardless of which
    extra audio packages were installed before the server booted.
    """
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not on PATH; needed to normalize audio for vLLM")
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(audio_path),
            "-vn", "-sn", "-dn",
            "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", "-f", "wav",
            "-",
        ],
        check=True, capture_output=True,
    )
    return proc.stdout


def transcribe_one(
    base_url: str,
    model: str,
    audio_path: Path,
    *,
    max_tokens: int = 600,
    temperature: float = 0.0,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 6,
    timeout: int = 600,
) -> dict:
    """Run the transcriber on one audio file. Returns dict with `text`,
    `parsed` (language/lyrics split), `seconds`, plus the raw response.

    `repetition_penalty` is a vLLM extension (>1.0 discourages repeats) —
    needed because the transcriber latches onto phrases like
    "[whispered] What is?" and loops them until max_tokens. Pass via
    `extra_body` so OpenAI-spec validators don't reject the request.
    """
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    # Always re-encode to WAV (16k mono PCM). Cheap, and dodges vLLM's mp3
    # codec dependency mess — soundfile handles WAV trivially.
    wav_bytes = _to_wav_bytes(audio_path)
    fmt = "wav"
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # vLLM extensions — passed at the top level for raw HTTP POST.
        # (The OpenAI Python SDK would wrap these in extra_body=…, but
        #  vLLM accepts both shapes; top-level is simpler here.)
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "messages": [
            {
                "role": "user",
                "content": [
                    _build_audio_part(audio_b64, fmt),
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    }
    t0 = time.time()
    r = requests.post(f"{base_url}/v1/chat/completions", json=body, timeout=timeout)
    seconds = round(time.time() - t0, 2)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"].strip()
    parsed = parse_transcription(text)
    # Clean the lyrics block before downstream use: strip pure-timecode
    # brackets and any stop-the-music markers ([abrupt silence], [song
    # ends], etc.). Runaway detection runs on the cleaned text so template
    # loops are still caught while structural noise gets eliminated outright.
    if parsed.get("lyrics"):
        parsed["lyrics"] = strip_silence_markers(strip_timecodes(parsed["lyrics"]))
    is_runaway, runaway_reason = looks_like_runaway(parsed.get("lyrics") or "")
    return {
        "text": text,
        "parsed": parsed,
        "seconds": seconds,
        "usage": data.get("usage"),
        "runaway": is_runaway,
        "runaway_reason": runaway_reason,
    }


_SECTION_RE = re.compile(r"^#\s*(\w[\w ]*)\s*$", re.MULTILINE)

# Pure timecode brackets — `[0:08]`, `[1:29]`, `[0:00-0:30]`, `[2:00 - 2:48]`.
# Strip these from transcriber output before anything downstream uses it.
# Brackets with extra content like `[0:00-0:30 Intro]` are left alone — those
# carry actual annotations the chain might want.
# Match all the timecode bracket shapes we've seen models emit:
#   [0:08]            mm:ss
#   [0:00-0:30]       mm:ss-mm:ss
#   [2:00 - 2:48]     mm:ss - mm:ss (with spaces)
#   [0:00-30]         mm:ss-secs (lyricist's invented format)
#   [1:00-120]        mm:ss-secs (longer secs)
# The `\d+(?::\d+)?` for the second half makes the `:\d+` optional, so
# both `[1:00-120]` and `[1:00-1:20]` match.
_TIMECODE_LEAD_RE = re.compile(
    r"^[ \t]*\[\s*\d+:\d+(?:\s*-\s*\d+(?::\d+)?)?\s*\][ \t]*",
    re.MULTILINE,
)
_TIMECODE_INLINE_RE = re.compile(
    r"\[\s*\d+:\d+(?:\s*-\s*\d+(?::\d+)?)?\s*\]\s*"
)

# Brackets that tell ACE "stop / cut / go silent" — descriptive in a real
# transcript but actively harmful when carried over as lyrics for the next
# cover-mode take (the DiT may literally render silence at that point).
# Matches things like [abrupt silence], [abrupt end], [Sudden silence],
# [song ends], [song ends abruptly], [end of track], [cut off].
_SILENCE_BRACKET_RE = re.compile(
    r"\[[^\]]*\b("
    r"silence|silent|abrupt|abruptly|"
    r"song\s+ends?|end\s+of\s+track|cut\s+off|cuts\s+off|stops?"
    r")\b[^\]]*\]\s*",
    re.IGNORECASE,
)


def strip_timecodes(text: str) -> str:
    """Remove pure `[mm:ss]` / `[mm:ss-mm:ss]` brackets from transcriber output.

    Leaves descriptive brackets like `[Melodic Theme 1: Violin]` untouched
    so chain training still gets the structural information. Collapses the
    extra blank lines the strip can leave behind.
    """
    if not text:
        return text
    out = _TIMECODE_LEAD_RE.sub("", text)
    out = _TIMECODE_INLINE_RE.sub("", out)
    # Collapse runs of blank lines created by the strip.
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def strip_silence_markers(text: str) -> str:
    """Remove stop-the-music brackets ([abrupt silence], [song ends], etc.)
    so they don't carry over as lyrics into the next cover-mode take.

    Useful descriptive brackets ([Final chord and fade out], [Outro - Sax
    Melody fades]) are kept — only the literal stop/silence/abrupt cluster
    is matched.
    """
    if not text:
        return text
    out = _SILENCE_BRACKET_RE.sub("", text)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def parse_transcription(text: str) -> dict:
    """Split the model output into its '# Languages' and '# Lyrics' sections."""
    sections: dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(text))
    for i, m in enumerate(matches):
        key = m.group(1).strip().lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[key] = text[start:end].strip()
    return {
        "languages": sections.get("languages"),
        "lyrics": sections.get("lyrics"),
    }


def looks_like_runaway(
    lyrics: str,
    *,
    max_template_repeat: int = 5,
    max_timestamp_lines: int = 10,
) -> tuple[bool, str]:
    """Detect when the transcriber has fallen into a templated-loop pattern.

    The runaway we keep hitting is bracketed structural annotations that
    increment a counter — `[Melodic Theme 1: Violin] [Melodic Theme 2: Violin]
    [Melodic Theme 3: Violin] …` — which `no_repeat_ngram_size` can't catch
    because each line *is* unique by its number, while the template clearly
    isn't musical content. We normalize digits to `N` and count repeats of
    the normalized bracket; if any template repeats more than
    `max_template_repeat` times we flag it.

    Also flags transcriptions that are mostly bare-timestamp anchors
    (`[0:08] [0:17] [0:34] …`), again because numbers vary but content
    is empty.

    Returns ``(is_runaway, reason)``. Empty/`[Instrumental]` is NOT a runaway.
    """
    if not lyrics:
        return False, ""
    text = lyrics.strip()
    if text.lower() == "[instrumental]" or len(text) < 20:
        return False, ""

    brackets = re.findall(r"\[[^\]]*\]", text)
    if not brackets:
        return False, ""

    # Normalize digits to 'N' so [Melodic Theme 1: Violin] and [Melodic Theme 2: Violin]
    # collapse to the same template.
    normalized: dict[str, int] = {}
    for b in brackets:
        norm = re.sub(r"\d+", "N", b)
        normalized[norm] = normalized.get(norm, 0) + 1
    worst = max(normalized.items(), key=lambda kv: kv[1])
    if worst[1] > max_template_repeat:
        return True, f"template {worst[0]!r} repeats {worst[1]}×"

    # Bare timestamp tokens like [0:08] or [1:29]
    bare_ts = sum(1 for b in brackets if re.match(r"^\[\s*\d+:\d+\s*\]$", b))
    if bare_ts > max_timestamp_lines:
        return True, f"{bare_ts} bare-timestamp anchors"

    return False, ""


def _print_result(audio: Path, result: dict) -> None:
    print(f"\n=== {audio} ({result['seconds']}s) ===")
    parsed = result["parsed"]
    if parsed.get("languages"):
        print(f"  language: {parsed['languages']}")
    if parsed.get("lyrics"):
        print("  --- lyrics ---")
        for line in parsed["lyrics"].splitlines():
            print(f"  {line}")
    else:
        print("  --- raw text ---")
        for line in result["text"].splitlines():
            print(f"  {line}")


def _append_jsonl(
    jsonl_path: Path,
    audio: Path,
    result: dict,
    extra: dict,
) -> None:
    record = {
        "source_audio": str(audio),
        "transcribed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "transcribe_seconds": result["seconds"],
        "raw_text": result["text"],
        "language": result["parsed"].get("languages"),
        "lyrics": result["parsed"].get("lyrics"),
        "model": DEFAULT_MODEL,
    }
    for k, v in extra.items():
        if v not in (None, ""):
            record[k] = v
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe audio with the ACE-Step transcriber via vLLM.",
    )
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--url", default=DEFAULT_BASE_URL,
                        help=f"vLLM base URL (default {DEFAULT_BASE_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"served model name (default {DEFAULT_MODEL})")
    parser.add_argument("--jsonl", type=Path,
                        help="append each result as a JSON line to this file "
                             "(default: state/transcribed.jsonl). Pass an explicit "
                             "path or '' to disable.")
    parser.add_argument("--no-save", action="store_true",
                        help="do not write JSONL at all")
    parser.add_argument("--artist", default=None)
    parser.add_argument("--album", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--track", default=None)
    parser.add_argument("--max-tokens", type=int, default=600,
                        help="generation cap (default 600; lower stops more runaway loops)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.15,
                        help="vLLM repetition penalty (>1.0 discourages loops; default 1.15)")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=6, dest="no_repeat_ngram_size",
                        help="ban any n-gram (length N) from repeating in the output (default 6)")
    parser.add_argument("--allow-runaway", action="store_true",
                        help="don't reject transcriptions detected as templated-loop runaway")
    parser.add_argument("--json", action="store_true",
                        help="print full JSON result instead of formatted output")
    args = parser.parse_args(argv)

    if not server_alive(args.url):
        print(
            f"transcriber not reachable at {args.url}. Start vLLM with:\n"
            f"  vllm serve {args.model} \\\n"
            f"      --host 0.0.0.0 --port 8000 \\\n"
            f"      --max-model-len 32768 --gpu-memory-utilization 0.9 \\\n"
            f"      --trust-remote-code",
            file=sys.stderr,
        )
        return 1

    paths = Paths.discover()
    jsonl_path: Path | None = None
    if not args.no_save:
        jsonl_path = args.jsonl or (paths.state_dir / "transcribed.jsonl")

    extra = {"artist": args.artist, "album": args.album,
             "title": args.title, "track": args.track}

    rc = 0
    for audio in args.audio:
        try:
            result = transcribe_one(
                args.url, args.model, audio,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
        except FileNotFoundError:
            print(f"error: {audio} does not exist", file=sys.stderr)
            rc = 1
            continue
        except requests.RequestException as exc:
            print(f"error: vLLM request failed for {audio}: {exc}", file=sys.stderr)
            rc = 1
            continue
        if result.get("runaway") and not args.allow_runaway:
            print(f"  REJECTED (runaway): {result.get('runaway_reason')}")
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            _print_result(audio, result)
        if jsonl_path is not None:
            _append_jsonl(jsonl_path, audio, result, extra)
            print(f"  appended JSON to {jsonl_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
