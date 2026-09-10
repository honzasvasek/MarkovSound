"""One generation cycle split into planning, lyrics, synthesis, output, and feedback phases."""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import ace_client, cli_transcribe, lyricist
from .codes_markov import format_codes, parse_codes, sample_codes, train_codes
from .config import AceConfig, Paths
from .composition import FormArc, choose_form, shape_caption
from .corpora import append_understood_lyrics, append_used_lyrics
from .describe import metadata_caption
from .latent_splice import frame_count, splice_latent_segments
from .lyrics_markov import sample_lyrics, train_lyrics, untrain_lyrics
from .markov import generate_caption, save_chain, train_text
from .runtime_config import RuntimeConfig
from .song import (
    _LM_CFG_SCALE,
    _LM_NEGATIVE_PROMPT,
    _LM_NEGATIVE_PROMPT_VOCAL,
    _experimental_metadata,
    _vocalize_caption,
)

_CODES_READY_THRESHOLD = 1500


@dataclass
class SongPlan:
    is_new_song: bool
    caption: str
    duration: int
    scaffold: dict[str, object]
    vocal_mode: bool
    prompt_caption: str
    form_arc: dict[str, str] | None = None


@dataclass
class LyricsPlan:
    request: dict
    source: str = "none"
    created_lyrics: str | None = None
    lyricist_seconds: float = 0.0


@dataclass
class SynthesisResult:
    enriched: dict
    codes_source: str
    lyrics_source: str
    lm_seconds: float
    mp3: bytes
    synth_seconds: float
    dead_end_positions: list[int] | None = None
    latent_splice_overlap_frames: list[int] | None = None


def lyric_token_bounds(duration: int) -> tuple[int, int]:
    max_tokens = max(150, duration * 2)
    min_tokens = max(40, max_tokens // 3)
    return min_tokens, max_tokens


def plan_song(
    *,
    chain: dict,
    order: int,
    paths: Paths,
    runtime_cfg: RuntimeConfig,
    requested_duration: int,
    song_state: dict | None,
    takes_per_song: int,
    random_duration_seconds: Callable[[RuntimeConfig], int],
    read_preset: Callable[[Paths], str],
    log: Callable[[str], None],
) -> SongPlan | None:
    is_new_song = song_state is None or song_state.get("takes_left", 0) <= 0 or not song_state.get("caption")
    if is_new_song:
        caption = generate_caption(
            chain,
            order,
            min_words=runtime_cfg.caption_min_words,
            max_words=runtime_cfg.caption_max_words,
            temperature=1.0,
        )
        if not caption:
            log("chain produced no caption; skipping cycle")
            return None
        preset = read_preset(paths)
        if preset:
            log(f"caption preset: {preset!r}")
            caption = f"{preset}. {caption}"
        duration = requested_duration if requested_duration > 0 else random_duration_seconds(runtime_cfg)
        if song_state is not None:
            song_state.update(
                caption=caption,
                duration=duration,
                takes_left=max(1, takes_per_song),
                take_no=0,
                last_mp3_path=None,
                last_transcribed_lyrics="",
                last_audio_codes="",
            )
        log(f"NEW SONG: caption={caption[:80]!r}{'...' if len(caption)>80 else ''}")
        log(f"          {duration}s, {song_state['takes_left'] if song_state else 1} takes")
    else:
        caption = song_state["caption"]
        duration = song_state["duration"]
        log(f"continuing SONG: take {song_state['take_no'] + 1} of {takes_per_song}")
        log(f"  caption={caption[:80]!r}{'...' if len(caption)>80 else ''}")
        log(f"  {duration}s")

    if song_state is not None:
        song_state["take_no"] += 1
        song_state["takes_left"] -= 1
        take_number = song_state["take_no"]
    else:
        take_number = 1
    if is_new_song:
        form = choose_form() if random.random() < runtime_cfg.form_arc_prob else None
        if song_state is not None:
            song_state["form_arc"] = form.to_dict() if form else None
    else:
        form = FormArc.from_dict(song_state.get("form_arc")) if song_state is not None else None
    prompt_caption = shape_caption(
        caption,
        form,
        take_number=take_number,
        total_takes=takes_per_song,
    )
    scaffold = _experimental_metadata()
    vocal_mode = random.random() < runtime_cfg.vocal_prob
    log(f"caption: {caption}")
    log(f"duration {duration}s / {duration / 60:.1f} min")
    log(f"scaffold: {scaffold['bpm']} bpm, {scaffold['keyscale']}, {scaffold['timesignature']}/4")
    if form is not None:
        log(f"form arc: {form.name} (take {take_number}/{takes_per_song})")
        log(f"prompt caption: {prompt_caption}")
    return SongPlan(
        is_new_song=is_new_song,
        caption=caption,
        duration=duration,
        scaffold=scaffold,
        vocal_mode=vocal_mode,
        prompt_caption=prompt_caption,
        form_arc=form.to_dict() if form else None,
    )


def prepare_lyrics(
    *,
    plan: SongPlan,
    lyrics_chain: dict,
    lyrics_order: int,
    ace_cfg: AceConfig,
    vocal_lang: str,
    log: Callable[[str], None],
) -> LyricsPlan:
    request = {
        "caption": plan.prompt_caption,
        "duration": plan.duration,
        "bpm": plan.scaffold["bpm"],
        "keyscale": plan.scaffold["keyscale"],
        "timesignature": plan.scaffold["timesignature"],
        "lm_cfg_scale": _LM_CFG_SCALE,
        "lm_negative_prompt": _LM_NEGATIVE_PROMPT_VOCAL if plan.vocal_mode else _LM_NEGATIVE_PROMPT,
        "use_cot_caption": False,
        "inference_steps": 8,
        "guidance_scale": 1.0,
        "shift": 3.0,
        "synth_model": ace_cfg.synth_model,
        "output_format": "mp3",
    }
    if not plan.vocal_mode:
        request["lyrics"] = "[Instrumental]"
        return LyricsPlan(request)

    vocal_caption = _vocalize_caption(plan.prompt_caption, vocal_lang)
    if vocal_caption != plan.prompt_caption:
        log(f"caption rewritten for vocal cycle: {vocal_caption[:160]}{'...' if len(vocal_caption) > 160 else ''}")
    request["caption"] = vocal_caption
    request["vocal_language"] = vocal_lang
    fragment, seconds = lyricist.generate_lyrics(
        vocal_caption,
        plan.duration,
        bpm=plan.scaffold["bpm"],
        description=plan.caption,
        max_tokens=400,
        log=log,
    )
    if fragment:
        added = train_lyrics(lyrics_chain, fragment, lyrics_order)
        log(f"lyricist→vocab: +{added} chain transitions from {len(fragment)} chars in {seconds}s")
    min_t, max_t = lyric_token_bounds(plan.duration)
    sampled = sample_lyrics(lyrics_chain, lyrics_order, min_tokens=min_t, max_tokens=max_t)
    if sampled:
        request["lyrics"] = sampled
        log(f"vocal cycle ({vocal_lang}); markov chain → {len(sampled)} chars (target {min_t}-{max_t}t)")
        return LyricsPlan(request, "markov", sampled, seconds)
    if fragment:
        request["lyrics"] = fragment
        log(f"vocal cycle ({vocal_lang}); chain too thin, using lyricist fragment directly ({len(fragment)} chars)")
        return LyricsPlan(request, "llama3-bootstrap", None, seconds)
    request["lyrics"] = ""
    log(f"vocal cycle ({vocal_lang}); lyricist + chain both empty — /lm will write lyrics")
    return LyricsPlan(request, "lm", None, seconds)


def synthesize(
    *,
    plan: SongPlan,
    lyrics_plan: LyricsPlan,
    codes_chain: dict,
    codes_order: int,
    ace_cfg: AceConfig,
    force_codes_mode: str,
    song_state: dict | None,
    latent_splice_seconds: float = 5.0,
    log: Callable[[str], None],
) -> SynthesisResult:
    request = lyrics_plan.request
    lyrics_source = lyrics_plan.source
    src_audio: Path | None = None
    codes_source = "lm"
    lm_seconds = 0.0
    dead_end_positions: list[int] | None = None
    use_cover = bool(song_state and not plan.is_new_song and song_state.get("last_mp3_path") and Path(song_state["last_mp3_path"]).exists())
    if use_cover:
        prev_mp3 = Path(song_state["last_mp3_path"])
        prev_lyrics = (song_state.get("last_transcribed_lyrics") or "").strip()
        if prev_lyrics and prev_lyrics != "[Instrumental]":
            request["lyrics"] = prev_lyrics
            lyrics_source = "carryover"
            log(f"cover take: reusing {len(prev_lyrics)}-char transcribed lyrics from take {song_state.get('take_no') - 1}")
        request["task_type"] = "cover"
        request["audio_cover_strength"] = song_state.get("cover_strength", 0.6)
        enriched = request
        src_audio = prev_mp3
        codes_source = "cover"
        log(f"cover mode: src={prev_mp3.name} strength={request['audio_cover_strength']}")
    else:
        use_markov = force_codes_mode == "markov" or (
            force_codes_mode == "auto" and sum(len(v) for v in codes_chain.values()) >= _CODES_READY_THRESHOLD
        )
        if force_codes_mode == "lm" or (plan.vocal_mode and use_markov):
            if plan.vocal_mode and use_markov:
                log("vocal cycle: routing through /lm so codes match the requested vocals")
            use_markov = False
        dead_end_positions: list[int] | None = None
        if use_markov and codes_chain:
            sampled, dead_end_positions = sample_codes(codes_chain, codes_order, max(1, int(plan.duration * 5)))
            log(f"sampled {len(sampled)} markov codes ({len(dead_end_positions)} dead-end jump(s))")
            request["audio_codes"] = format_codes(sampled)
            request["vocal_language"] = "en"
            enriched = request
            codes_source = "markov"
        else:
            if use_markov and not codes_chain:
                log("markov codes requested but chain is empty; falling back to /lm")
            t0 = time.time()
            log(f"SENDING TO /LM: {json.dumps(request, indent=2)}")
            enriched = ace_client.lm(ace_cfg, request, log=log)
            lm_seconds = round(time.time() - t0, 2)
            log(f"/lm done in {lm_seconds:.1f}s (bpm={enriched.get('bpm')} key={enriched.get('keyscale')})")
            if plan.vocal_mode:
                generated = enriched.get("lyrics") or ""
                log(f"LM generated lyrics: {generated[:100]!r}{'...' if len(generated) > 100 else ''}")
    t0 = time.time()
    splice_overlaps: list[int] | None = None
    if codes_source == "markov" and dead_end_positions and latent_splice_seconds > 0:
        boundaries = [p for p in dead_end_positions if 0 < p < len(sampled)]
        parts = [sampled[a:b] for a, b in zip([0, *boundaries], [*boundaries, len(sampled)])]
        segment_latents: list[bytes] = []
        for index, part in enumerate(parts, start=1):
            segment_request = dict(enriched)
            segment_request["audio_codes"] = format_codes(part)
            segment_request["duration"] = len(part) / 5.0
            _segment_mp3, segment_latent = ace_client.synth(ace_cfg, segment_request, log=log)
            if segment_latent is None:
                log("latent splice unavailable: /synth returned no latent; falling back to direct synth")
                segment_latents = []
                break
            segment_latents.append(segment_latent)
            log(f"latent splice segment {index}/{len(parts)}: {len(part)} codes, {frame_count(segment_latent)} frames")
        if segment_latents:
            merged_latents, splice_overlaps = splice_latent_segments(segment_latents, latent_splice_seconds)
            mp3 = ace_client.vae_decode(ace_cfg, merged_latents, log=log)
            enriched["duration"] = frame_count(merged_latents) / 25.0
            log(f"latent-spliced {len(parts)} segments with overlap frames={splice_overlaps}")
        else:
            mp3, _latent = ace_client.synth(ace_cfg, enriched, src_audio=src_audio, log=log)
    else:
        mp3, _latent = ace_client.synth(ace_cfg, enriched, src_audio=src_audio, log=log)
    synth_seconds = round(time.time() - t0, 2)
    log(f"/synth done in {synth_seconds:.1f}s (mp3={len(mp3)} bytes, codes={codes_source})")
    return SynthesisResult(enriched, codes_source, lyrics_source, lm_seconds, mp3, synth_seconds,
                           dead_end_positions, splice_overlaps)


def write_output(
    *, cycle: int, plan: SongPlan, lyrics_plan: LyricsPlan, result: SynthesisResult,
    paths: Paths, song_state: dict | None,
) -> tuple[Path, Path, dict]:
    paths.staging_dir.mkdir(parents=True, exist_ok=True)
    base = f"{cycle:06d}"
    mp3_path = paths.staging_dir / f"{base}.mp3"
    json_path = paths.staging_dir / f"{base}.json"
    # Keep work-in-progress outputs out of Audio/queue until feedback has
    # finished, so ./play only ever sees complete mp3/json pairs.
    mp3_path.write_bytes(result.mp3)
    append_used_lyrics(paths, cycle=cycle, mp3_name=mp3_path.name, enriched=result.enriched,
                       duration=result.enriched.get("duration"), vocal_mode=plan.vocal_mode,
                       lyrics_source=result.lyrics_source)
    sidecar = {
        "cycle": cycle, "caption": plan.caption,
        "prompt_caption": plan.prompt_caption, "form_arc": plan.form_arc,
        "enriched_caption": result.enriched.get("caption"), "scaffold": plan.scaffold,
        "codes_source": result.codes_source, "vocal_mode": plan.vocal_mode,
        "lyrics_source": result.lyrics_source, "created_lyrics": lyrics_plan.created_lyrics,
        "used_lyrics": result.enriched.get("lyrics"), "bpm": result.enriched.get("bpm"),
        "keyscale": result.enriched.get("keyscale"), "timesignature": result.enriched.get("timesignature"),
        "vocal_language": result.enriched.get("vocal_language"), "duration": result.enriched.get("duration"),
        "seed": result.enriched.get("seed"), "lm_seed": result.enriched.get("lm_seed"),
        "lyrics": result.enriched.get("lyrics"), "lm_model": result.enriched.get("lm_model"),
        "synth_model": result.enriched.get("synth_model"), "inference_steps": result.enriched.get("inference_steps"),
        "lm_seconds": result.lm_seconds, "lyricist_seconds": lyrics_plan.lyricist_seconds,
        "synth_seconds": result.synth_seconds, "audio_codes": result.enriched.get("audio_codes"),
        "dead_end_jump_code_positions": result.dead_end_positions,
        "latent_splice_overlap_frames": result.latent_splice_overlap_frames,
        "trained_caption": None, "code_count": None,
    }
    return mp3_path, json_path, sidecar



def publish_output(
    *, mp3_path: Path, json_path: Path, sidecar: dict, paths: Paths,
) -> tuple[Path, Path]:
    """Move a complete staged track pair into the playback queue."""
    json_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n")
    paths.queue_dir.mkdir(parents=True, exist_ok=True)
    queued_mp3 = paths.queue_dir / mp3_path.name
    queued_json = paths.queue_dir / json_path.name
    json_path.replace(queued_json)
    mp3_path.replace(queued_mp3)
    return queued_mp3, queued_json

def train_generated(
    *, result: SynthesisResult, lyrics_plan: LyricsPlan, sidecar: dict,
    codes_chain: dict, codes_order: int, lyrics_chain: dict, lyrics_order: int,
    vocal_mode: bool, log: Callable[[str], None],
) -> None:
    if result.codes_source == "lm":
        seq = parse_codes(result.enriched.get("audio_codes") or "")
        if seq:
            added = train_codes(codes_chain, seq, codes_order)
            sidecar["code_count"] = len(seq)
            log(f"trained codes chain (+{added} from /lm; len={len(seq)})")
    lm_lyrics = result.enriched.get("lyrics") or ""
    if vocal_mode and lm_lyrics and lm_lyrics.strip() != "[Instrumental]":
        if result.lyrics_source == "lm":
            added = train_lyrics(lyrics_chain, lm_lyrics, lyrics_order)
            log(f"trained lyrics chain (+{added} from /lm)")
        elif result.lyrics_source == "markov+format" and lyrics_plan.created_lyrics and lm_lyrics != lyrics_plan.created_lyrics:
            removed = untrain_lyrics(lyrics_chain, lyrics_plan.created_lyrics, lyrics_order)
            added = train_lyrics(lyrics_chain, lm_lyrics, lyrics_order)
            log(f"replaced created lyrics in chain (-{removed} raw, +{added} /lm-formatted)")



@dataclass
class TranscriptionResult:
    lyrics: str = ""
    language: str = ""


def _understand_track(*, ace_cfg: AceConfig, mp3_path: Path, sidecar: dict, log: Callable[[str], None]) -> tuple[dict, str | None]:
    """Run ACE understand and record the machine-readable observation block."""
    t0 = time.time()
    meta, _latent = ace_client.understand(ace_cfg, mp3_path, log=log)
    sidecar["understand_seconds"] = round(time.time() - t0, 2)
    sidecar["understand_meta"] = {
        key: meta.get(key)
        for key in ("caption", "bpm", "keyscale", "timesignature", "vocal_language", "duration")
    }
    return meta, metadata_caption(meta)


def _train_understood_codes(*, meta: dict, codes_chain: dict, codes_order: int, log: Callable[[str], None]) -> None:
    """Teach the codes chain from what ACE heard, independent of text feedback."""
    seq = parse_codes(meta.get("audio_codes") or "")
    if seq:
        added = train_codes(codes_chain, seq, codes_order)
        log(f"trained codes chain (+{added} from /understand; len={len(seq)})")


def _transcribe_track(*, mp3_path: Path, sidecar: dict, log: Callable[[str], None]) -> TranscriptionResult:
    """Prefer the richer external transcriber, but degrade cleanly when unavailable."""
    base_url = cli_transcribe.DEFAULT_BASE_URL
    try:
        if not cli_transcribe.server_alive(base_url):
            return TranscriptionResult()
        woke = cli_transcribe.wake(base_url)
        if woke:
            log("transcriber: woke from sleep")
        t0 = time.time()
        result = cli_transcribe.transcribe_one(
            base_url,
            cli_transcribe.DEFAULT_MODEL,
            mp3_path,
        )
        raw = (result["parsed"].get("lyrics") or "").strip()
        language = (result["parsed"].get("languages") or "").strip()
        sidecar["transcriber_lang"] = language
        sidecar["transcriber_seconds"] = round(time.time() - t0, 2)
        if result.get("runaway"):
            # Some transcriber failures look like long, plausible text. Keep
            # them for inspection, but never train/carry them forward.
            sidecar["transcriber_lyrics_raw"] = raw
            sidecar["transcriber_runaway"] = result.get("runaway_reason")
            log(f"transcriber: REJECTED ({result.get('runaway_reason')})")
            return TranscriptionResult(language=language)
        sidecar["transcriber_lyrics"] = raw
        log(f"transcriber: {len(raw)} chars, lang={language!r}")
        return TranscriptionResult(raw, language)
    except Exception as exc:  # noqa: BLE001
        log(f"transcriber failed (continuing with /understand lyrics): {exc!r}")
        return TranscriptionResult()
    finally:
        try:
            if cli_transcribe.sleep(base_url):
                log("transcriber: sleeping")
        except Exception as exc:  # noqa: BLE001
            log(f"transcriber sleep failed: {exc!r}")


def _train_heard_lyrics(
    *, plan: SongPlan, meta: dict, transcription: TranscriptionResult,
    lyrics_chain: dict, lyrics_order: int, log: Callable[[str], None],
) -> str:
    """Choose the best heard lyrics source and feed it into the lyrics chain."""
    heard = transcription.lyrics or (meta.get("lyrics") or "")
    if plan.vocal_mode and heard and heard.strip() != "[Instrumental]":
        added = train_lyrics(lyrics_chain, heard, lyrics_order)
        source = "transcriber" if transcription.lyrics else "/understand"
        log(f"trained lyrics chain (+{added} from {source})")
    return heard


def _store_take_carryover(*, song_state: dict | None, meta: dict, transcription: TranscriptionResult) -> None:
    """Persist material the next cover take should inherit from this take."""
    if song_state is not None and song_state.get("takes_left", 0) > 0:
        song_state["last_transcribed_lyrics"] = transcription.lyrics or (meta.get("lyrics") or "")
        song_state["last_audio_codes"] = meta.get("audio_codes") or ""


def _append_feedback_corpus(
    *, paths: Paths, cycle: int, mp3_path: Path, meta: dict, heard_lyrics: str,
    transcription: TranscriptionResult, plan: SongPlan, result: SynthesisResult,
) -> None:
    """Write the exact lyrics source used for feedback into the audit corpus."""
    corpus_meta = dict(meta)
    corpus_meta["lyrics"] = heard_lyrics
    if transcription.language:
        corpus_meta["vocal_language"] = transcription.language
    append_understood_lyrics(
        paths,
        cycle=cycle,
        mp3_name=mp3_path.name,
        meta=corpus_meta,
        duration=result.enriched.get("duration"),
        vocal_mode=plan.vocal_mode,
        lyrics_source=f"transcriber+{result.lyrics_source}" if transcription.lyrics else result.lyrics_source,
    )


def _train_understood_caption(
    *, trained_caption: str | None, chain: dict, order: int, paths: Paths,
    sidecar: dict, log: Callable[[str], None],
) -> None:
    """Train the text chain only when understand returned a usable caption."""
    if not trained_caption:
        log("understand returned no usable caption; skipping text train")
        return
    added = train_text(chain, trained_caption, order)
    save_chain(chain, order, paths.chain_path)
    sidecar["trained_caption"] = trained_caption
    log(f"trained text chain (+{added}): {trained_caption[:160]}{'...' if len(trained_caption) > 160 else ''}")

def apply_feedback(
    *, ace_cfg: AceConfig, mp3_path: Path, json_path: Path, sidecar: dict,
    chain: dict, order: int, codes_chain: dict, codes_order: int,
    lyrics_chain: dict, lyrics_order: int, paths: Paths, cycle: int,
    plan: SongPlan, result: SynthesisResult, song_state: dict | None,
    log: Callable[[str], None],
) -> None:
    """Observe the generated track, learn from it, and persist feedback metadata."""
    meta, trained_caption = _understand_track(
        ace_cfg=ace_cfg, mp3_path=mp3_path, sidecar=sidecar, log=log,
    )
    _train_understood_codes(meta=meta, codes_chain=codes_chain, codes_order=codes_order, log=log)
    transcription = _transcribe_track(mp3_path=mp3_path, sidecar=sidecar, log=log)
    _store_take_carryover(song_state=song_state, meta=meta, transcription=transcription)
    heard_lyrics = _train_heard_lyrics(
        plan=plan, meta=meta, transcription=transcription,
        lyrics_chain=lyrics_chain, lyrics_order=lyrics_order, log=log,
    )
    _append_feedback_corpus(
        paths=paths, cycle=cycle, mp3_path=mp3_path, meta=meta,
        heard_lyrics=heard_lyrics, transcription=transcription, plan=plan, result=result,
    )
    _train_understood_caption(
        trained_caption=trained_caption, chain=chain, order=order,
        paths=paths, sidecar=sidecar, log=log,
    )
    json_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n")
