"""Low-level client for ACE-Step server lifecycle and job endpoints."""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from email.parser import BytesParser
from email.policy import default as email_default_policy
from pathlib import Path
from typing import Any

import requests

from .config import AceConfig


_POLL_SECONDS = 1.5
_POLL_TIMEOUT = 60 * 30  # 30 minutes for a long synth


class AceError(RuntimeError):
    pass


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            sock.connect((host, port))
        except OSError:
            return False
        return True


def server_alive(cfg: AceConfig) -> bool:
    if not _port_open(cfg.host, cfg.port):
        return False
    try:
        r = requests.get(f"{cfg.base_url}/health", timeout=3)
        return r.ok and r.json().get("status") == "ok"
    except (requests.RequestException, ValueError):
        return False


def ensure_server(cfg: AceConfig, log=print) -> subprocess.Popen | None:
    """Start ace-server if it's not already running. Return the Popen we spawned
    (or None if a server was already up). Caller is responsible for tearing
    down the returned process.
    """
    if server_alive(cfg):
        log(f"ace-server already running on {cfg.base_url}")
        return None
    if not cfg.server_bin.exists():
        raise AceError(f"ace-server binary not found at {cfg.server_bin}")
    if not cfg.models_dir.exists():
        raise AceError(f"ace-server models dir not found at {cfg.models_dir}")
    log(f"starting ace-server on {cfg.base_url} (models={cfg.models_dir})")
    log_path = cfg.ace_root / "ace-server.log"
    log_handle = log_path.open("ab", buffering=0)
    proc = subprocess.Popen(
        [
            str(cfg.server_bin),
            "--host", cfg.host,
            "--port", str(cfg.port),
            "--models", str(cfg.models_dir),
            "--max-batch", "1",
        ],
        cwd=str(cfg.ace_root),
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.time() + 60
    while time.time() < deadline:
        if server_alive(cfg):
            log(f"ace-server ready (pid={proc.pid}, log={log_path})")
            return proc
        if proc.poll() is not None:
            raise AceError(f"ace-server exited with code {proc.returncode}; see {log_path}")
        time.sleep(0.5)
    proc.terminate()
    raise AceError(f"ace-server did not become ready within 60s; see {log_path}")


def shutdown_server(proc: subprocess.Popen | None, log=print) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    log(f"stopping ace-server (pid={proc.pid})")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _submit_json(cfg: AceConfig, endpoint: str, payload: dict[str, Any]) -> str:
    r = requests.post(f"{cfg.base_url}{endpoint}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["id"]


def _submit_multipart(cfg: AceConfig, endpoint: str, files: dict[str, tuple]) -> str:
    r = requests.post(f"{cfg.base_url}{endpoint}", files=files, timeout=120)
    r.raise_for_status()
    return r.json()["id"]


def _wait_job(cfg: AceConfig, job_id: str, log=print) -> None:
    deadline = time.time() + _POLL_TIMEOUT
    last_status = ""
    while time.time() < deadline:
        r = requests.get(f"{cfg.base_url}/job", params={"id": job_id}, timeout=10)
        r.raise_for_status()
        status = r.json().get("status", "")
        if status != last_status:
            log(f"  job {job_id}: {status}")
            last_status = status
        if status == "done":
            return
        if status in ("failed", "cancelled"):
            raise AceError(f"job {job_id} ended with status={status}")
        time.sleep(_POLL_SECONDS)
    raise AceError(f"job {job_id} timed out after {_POLL_TIMEOUT}s")


def _fetch_result(cfg: AceConfig, job_id: str) -> requests.Response:
    r = requests.get(
        f"{cfg.base_url}/job",
        params={"id": job_id, "result": "1"},
        timeout=60,
        stream=False,
    )
    r.raise_for_status()
    return r


def _parse_multipart(body: bytes, content_type: str) -> list[tuple[dict[str, str], bytes]]:
    """Return list of (headers_dict, payload_bytes). Uses stdlib email parser."""
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
    msg = BytesParser(policy=email_default_policy).parsebytes(header + body)
    out: list[tuple[dict[str, str], bytes]] = []
    for part in msg.iter_parts():
        headers = {k.lower(): v for k, v in part.items()}
        payload = part.get_payload(decode=True) or b""
        out.append((headers, payload))
    return out


def lm(cfg: AceConfig, request: dict[str, Any], log=print) -> dict[str, Any]:
    """Run /lm and return the (single) enriched AceRequest JSON."""
    payload = dict(request)
    payload.setdefault("lm_model", cfg.lm_model)
    payload.setdefault("synth_model", cfg.synth_model)
    log(f"POST /lm caption={payload.get('caption', '')[:80]!r}")
    job_id = _submit_json(cfg, "/lm", payload)
    _wait_job(cfg, job_id, log=log)
    r = _fetch_result(cfg, job_id)
    results = r.json()
    if not isinstance(results, list) or not results:
        raise AceError(f"/lm returned unexpected JSON: {results!r}")
    return results[0]


_INSTRUMENTAL_BRACKET_RE = re.compile(
    r"\[[^\]]*\binstrumental\b[^\]]*\]",
    re.IGNORECASE,
)


def _scrub_for_format(lyrics: str) -> str:
    """Drop literal [Instrumental*] brackets before sending to lm_mode=format.

    The chain produces a lot of `[Instrumental Section X]` / `[Outro -
    Instrumental]` / `[Chorus - Instrumental]` markers. The Qwen LM behind
    /lm sees those and collapses the whole piece to `[Instrumental]`, which
    silences the cycle. Stripping just the literal-instrumental brackets
    keeps every other Zappa stage direction intact for the format pass.
    """
    out = _INSTRUMENTAL_BRACKET_RE.sub("", lyrics)
    # Collapse runs of blank lines the strip can leave behind.
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def format_lyrics(
    cfg: AceConfig,
    caption: str,
    lyrics: str,
    vocal_language: str = "en",
    extra: dict[str, Any] | None = None,
    log=print,
) -> str:
    """Polish/structure user-supplied lyrics via /lm with lm_mode='format'.

    Per docs/ARCHITECTURE.md: 'format' takes caption + lyrics and returns
    metadata + lyrics, no codes — so it's a cheap LM pass that reshapes our
    chain output into something ACE will sing more confidently. Returns the
    polished lyrics string (falls back to the input on failure).
    """
    scrubbed = _scrub_for_format(lyrics)
    if not scrubbed:
        log("format pass: nothing left after scrubbing instrumental brackets")
        return lyrics
    payload: dict[str, Any] = {
        "caption": caption,
        "lyrics": scrubbed,
        "vocal_language": vocal_language,
        "lm_mode": "format",
    }
    if extra:
        payload.update(extra)
    log(f"POST /lm lm_mode=format lyrics_len={len(scrubbed)} (scrubbed from {len(lyrics)})")
    try:
        result = lm(cfg, payload, log=log)
    except AceError as exc:
        log(f"format pass failed ({exc!r}); keeping chain lyrics as-is")
        return lyrics
    polished = (result.get("lyrics") or "").strip()
    if not polished:
        log("format pass returned empty lyrics; keeping chain lyrics as-is")
        return lyrics
    if polished.lower() in ("[instrumental]", "instrumental", "") or len(polished) < max(40, len(lyrics) // 10):
        log(f"format pass collapsed ({len(lyrics)}→{len(polished)} chars, polished={polished!r}); keeping chain lyrics")
        return lyrics
    # Runaway guard: the LM sometimes echoes/expands a passage 50–100×
    # ([Male Voice] One, two, three, … on repeat). If any bracket or full
    # line appears more than 8× in the polished output, treat the polish
    # as poisoned and keep the chain output.
    bracket_counter: dict[str, int] = {}
    for m in re.finditer(r"\[[^\]]*\]", polished):
        bracket_counter[m.group(0)] = bracket_counter.get(m.group(0), 0) + 1
    line_counter: dict[str, int] = {}
    for ln in polished.splitlines():
        s = ln.strip()
        if s:
            line_counter[s] = line_counter.get(s, 0) + 1
    max_bracket = max(bracket_counter.values(), default=0)
    max_line = max(line_counter.values(), default=0)
    if max_bracket > 8 or max_line > 8:
        worst_b = max(bracket_counter, key=bracket_counter.get, default="-")
        worst_l = max(line_counter, key=line_counter.get, default="-")
        log(
            f"format pass produced runaway "
            f"(max bracket={max_bracket} {worst_b!r}, max line={max_line} {worst_l!r}); "
            "keeping chain lyrics"
        )
        return lyrics
    return polished


def synth(
    cfg: AceConfig,
    request: dict[str, Any],
    *,
    src_audio: Path | None = None,
    log=print,
) -> tuple[bytes, bytes | None]:
    """Run /synth from an LM-enriched AceRequest. Return (mp3_bytes, latent_bytes_or_None).

    When `src_audio` is provided, /synth runs in multipart mode: the JSON
    request travels as the 'request' part and the audio file as the 'audio'
    part. Used by cover/repaint/lego task types where ACE needs a source
    track to transform.
    """
    payload = dict(request)
    payload.setdefault("synth_model", cfg.synth_model)
    payload.setdefault("output_format", "mp3")
    log(
        f"POST /synth duration={payload.get('duration')} bpm={payload.get('bpm')}"
        + (f" src_audio={Path(src_audio).name} task={payload.get('task_type')}" if src_audio else "")
    )
    if src_audio is None:
        job_id = _submit_json(cfg, "/synth", payload)
    else:
        src = Path(src_audio)
        files = {
            "request": (None, json.dumps(payload), "application/json"),
            "audio": (src.name, src.read_bytes(), "application/octet-stream"),
        }
        r = requests.post(f"{cfg.base_url}/synth", files=files, timeout=120)
        r.raise_for_status()
        job_id = r.json()["id"]
    _wait_job(cfg, job_id, log=log)
    r = _fetch_result(cfg, job_id)
    ctype = r.headers.get("Content-Type", "")
    if ctype.startswith("multipart/"):
        parts = _parse_multipart(r.content, ctype)
        mp3 = None
        latent = None
        for headers, body in parts:
            sub_ctype = headers.get("content-type", "")
            if sub_ctype.startswith("audio/") and mp3 is None:
                mp3 = body
            elif sub_ctype.startswith("application/octet-stream") and latent is None:
                latent = body
        if mp3 is None:
            raise AceError(f"/synth multipart missing audio part (got {len(parts)} parts)")
        return mp3, latent
    if ctype.startswith("audio/"):
        return r.content, None
    raise AceError(f"/synth returned unexpected Content-Type: {ctype}")


def understand(cfg: AceConfig, audio_path: Path, log=print) -> tuple[dict[str, Any], bytes | None]:
    """Run /understand on an audio file. Return (metadata_dict, latent_bytes_or_None)."""
    log(f"POST /understand {audio_path.name}")
    with audio_path.open("rb") as fh:
        files = {"audio": (audio_path.name, fh, "application/octet-stream")}
        job_id = _submit_multipart(cfg, "/understand", files)
    _wait_job(cfg, job_id, log=log)
    r = _fetch_result(cfg, job_id)
    ctype = r.headers.get("Content-Type", "")
    if ctype.startswith("multipart/"):
        parts = _parse_multipart(r.content, ctype)
        meta = None
        latent = None
        for headers, body in parts:
            sub_ctype = headers.get("content-type", "")
            if sub_ctype.startswith("application/json") and meta is None:
                meta = json.loads(body.decode("utf-8"))
            elif sub_ctype.startswith("application/octet-stream") and latent is None:
                latent = body
        if meta is None:
            raise AceError(f"/understand multipart missing JSON part (got {len(parts)} parts)")
        if isinstance(meta, list):
            if not meta:
                raise AceError("/understand returned an empty JSON list")
            meta = meta[0]
        return meta, latent
    if ctype.startswith("application/json"):
        meta = r.json()
        if isinstance(meta, list):
            if not meta:
                raise AceError("/understand returned an empty JSON list")
            meta = meta[0]
        return meta, None
    raise AceError(f"/understand returned unexpected Content-Type: {ctype}")


def vae_decode(cfg: AceConfig, latents: bytes, *, log=print) -> bytes:
    """Decode raw flat `[T,64]` f32 latent bytes through ACE `/vae`."""
    log(f"POST /vae decode latent_bytes={len(latents)}")
    # `/vae` accepts exactly one payload side: audio for encode or src_latents
    # for decode. Decode uses the server's configured default output format.
    files = {
        "src_latents": ("splice.vae", latents, "application/octet-stream"),
    }
    job_id = _submit_multipart(cfg, "/vae", files)
    _wait_job(cfg, job_id, log=log)
    r = _fetch_result(cfg, job_id)
    ctype = r.headers.get("Content-Type", "")
    if ctype.startswith("audio/"):
        return r.content
    raise AceError(f"/vae decode returned unexpected Content-Type: {ctype}")
