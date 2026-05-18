from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import sys
import time
from pathlib import Path

from . import ace_client
from . import cli_transcribe
from . import lyricist
from .codes_markov import (
    build_codes_chain,
    chain_stats as codes_chain_stats,
    format_codes,
    load_codes_chain,
    parse_codes,
    sample_codes,
    save_codes_chain,
    train_codes,
)
from .config import AceConfig, Paths
from .corpora import append_understood_lyrics, append_used_lyrics
from .playback import wait_for_player
from .runtime_config import RuntimeConfig, load_runtime_config
from .song import (
    _LM_CFG_SCALE,
    _LM_NEGATIVE_PROMPT,
    _LM_NEGATIVE_PROMPT_VOCAL,
    _experimental_metadata,
    _vocalize_caption,
)
from .describe import is_too_pop, metadata_caption
from .lyrics_markov import (
    chain_stats as lyrics_chain_stats,
    load_lyrics_chain,
    sample_lyrics,
    save_lyrics_chain,
    train_lyrics,
    untrain_lyrics,
)
from .steering import load_allowed
from .markov import (
    build_chain,
    generate_caption,
    load_chain,
    save_chain,
    train_text,
)


# Below this many transitions the codes chain isn't varied enough yet —
# fall back to /lm. With order-1 over ~64k vocab a single /understand call
# adds ~150 transitions; 1500 ≈ 10 absorbed/generated tracks.
_CODES_READY_THRESHOLD = 1500
_CODES_ORDER = 3
_LYRICS_READY_THRESHOLD = 2000
_LYRICS_ORDER = 3


_DEFAULT_ORDER = 3
_DEFAULT_DURATION = 0  # 0 = random duration each cycle
_DEFAULT_MIN_WORDS = 18
_DEFAULT_MAX_WORDS = 70
_SLOT_COUNT = 10


def _random_duration_seconds(cfg: RuntimeConfig) -> int:
    minutes = cfg.duration_minutes()
    # Preserve a short-track bias while allowing the range to change live.
    weights = list(range(len(minutes), 0, -1))
    return random.choices(minutes, weights=weights, k=1)[0] * 60


def _pick_slot() -> int:
    return random.randint(1, _SLOT_COUNT)


def _lyric_token_bounds(duration: int) -> tuple[int, int]:
    """Min/max lyric token counts for a track of `duration` seconds.

    Scales roughly linearly: ~2 tokens/sec at the upper bound, with a hard
    floor for very short pieces. A 1-min track samples ~60–150 tokens; a
    10-min track ~400–1200. Tokens include bracket headers + word tokens
    + newlines as counted by lyrics_markov.tokenize_lyrics.
    """
    max_tokens = max(150, duration * 2)
    min_tokens = max(40, max_tokens // 3)
    return min_tokens, max_tokens

def log(msg: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    sys.stdout.write(f"[{timestamp}] {msg}\n")
    sys.stdout.flush()


def _slugify(text: str, limit: int = 48) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:limit] or "track"


def _load_or_build_chain(paths: Paths, order: int) -> tuple[dict, int]:
    if paths.chain_path.exists():
        chain, loaded_order = load_chain(paths.chain_path)
        log(f"loaded chain: {len(chain)} states, order={loaded_order}")
        return chain, loaded_order
    if not paths.corpus_path.exists():
        raise SystemExit(f"no chain and no corpus at {paths.corpus_path}")
    corpus = [
        line.strip() for line in paths.corpus_path.read_text().splitlines() if line.strip()
    ]
    log(f"building chain from {len(corpus)} corpus lines (order={order})")
    chain = build_chain(corpus, order)
    save_chain(chain, order, paths.chain_path)
    log(f"saved chain: {len(chain)} states")
    return chain, order


def _load_codes_chain(paths: Paths) -> tuple[dict, int]:
    if paths.codes_chain_path.exists():
        chain, order = load_codes_chain(paths.codes_chain_path)
        stats = codes_chain_stats(chain)
        log(f"loaded codes chain: {stats}")
        return chain, order
    log(f"no codes chain yet (will build as we go)")
    return {}, _CODES_ORDER


def _load_lyrics_chain(paths: Paths) -> tuple[dict, int]:
    if paths.lyrics_chain_path.exists():
        chain, order = load_lyrics_chain(paths.lyrics_chain_path)
        stats = lyrics_chain_stats(chain)
        log(f"loaded lyrics chain: {stats}")
        return chain, order
    log(f"no lyrics chain yet (will build as we go)")
    return {}, _LYRICS_ORDER


def _read_cycle(paths: Paths) -> int:
    try:
        return int(paths.cycle_path.read_text().strip())
    except (OSError, ValueError):
        return 0


def _write_cycle(paths: Paths, cycle: int) -> None:
    paths.cycle_path.parent.mkdir(parents=True, exist_ok=True)
    paths.cycle_path.write_text(f"{cycle}\n")


def _ensure_audio_dir(audio_dir: Path) -> None:
    audio_dir.mkdir(parents=True, exist_ok=True)


def _read_preset(paths: Paths) -> str:
    """Optional caption preset prepended to every fresh markov caption.

    Stored at state/preset.txt; re-read each cycle so it can be edited
    live without restarting ./create. Empty file or missing file = no
    preset.
    """
    try:
        text = paths.preset_path.read_text().strip()
    except (OSError, FileNotFoundError):
        return ""
    return text


def _run_cycle(
    cycle: int,
    chain: dict,
    order: int,
    codes_chain: dict,
    codes_order: int,
    lyrics_chain: dict,
    lyrics_order: int,
    paths: Paths,
    ace_cfg: AceConfig,
    runtime_cfg: RuntimeConfig,
    duration: int,  # 0 = pick randomly per cycle
    autofeedback: bool,
    force_codes_mode: str = "auto",  # "auto" | "markov" | "lm"
    vocal_prob: float = 0.0,
    vocal_lang: str = "en",
    force_lyrics_mode: str = "auto",  # "auto" | "markov" | "lm"
    song_state: dict | None = None,
    takes_per_song: int = 1,
) -> bool:
    log(f"=== cycle {cycle} ===")
    # Song concept: a human composer plays the same song several times,
    # evolving it along the way. Take 1 picks fresh caption/slot/duration;
    # takes 2..N reuse the slot + caption (so the player, which follows
    # newest-by-mtime, hears the same song re-rendered) but get a fresh
    # bpm/key/timesig scaffold and a new random seed via ACE, so each take
    # is a real performance variation rather than a bit-identical replay.
    is_new_song = (
        song_state is None
        or song_state.get("takes_left", 0) <= 0
        or not song_state.get("caption")
    )
    if is_new_song:
        caption = generate_caption(
            chain, order,
            min_words=runtime_cfg.caption_min_words,
            max_words=runtime_cfg.caption_max_words,
            temperature=1.0,
        )
        if not caption:
            log("chain produced no caption; skipping cycle")
            return False
        preset = _read_preset(paths)
        if preset:
            log(f"caption preset: {preset!r}")
            caption = f"{preset}. {caption}"
        slot = _pick_slot()
        song_duration = duration if duration > 0 else _random_duration_seconds(runtime_cfg)
        if song_state is not None:
            song_state["caption"] = caption
            song_state["slot"] = slot
            song_state["duration"] = song_duration
            song_state["takes_left"] = max(1, takes_per_song)
            song_state["take_no"] = 0
            # Clear carry-overs from previous song so take 1 of the new song
            # never accidentally inherits another composition's mp3/lyrics.
            song_state["last_mp3_path"] = None
            song_state["last_transcribed_lyrics"] = ""
            song_state["last_audio_codes"] = ""
        log(f"NEW SONG: caption={caption[:80]!r}{'...' if len(caption)>80 else ''}")
        log(f"          slot {slot:02d}, {song_duration}s, {song_state['takes_left'] if song_state else 1} takes")
    else:
        caption = song_state["caption"]
        slot = song_state["slot"]
        song_duration = song_state["duration"]
        log(f"continuing SONG: take {song_state['take_no'] + 1} of {takes_per_song}")
        log(f"  caption={caption[:80]!r}{'...' if len(caption)>80 else ''}")
        log(f"  slot {slot:02d}, {song_duration}s")

    if song_state is not None:
        song_state["take_no"] += 1
        song_state["takes_left"] -= 1

    log(f"caption: {caption}")
    duration = song_duration
    log(f"slot {slot:02d} ({duration}s / {duration / 60:.1f} min)")

    # Fresh scaffold per take so each rendition has its own bpm/key/timesig
    # — that's the per-take "variation". Caption stays the same.
    scaffold = _experimental_metadata()
    log(f"scaffold: {scaffold['bpm']} bpm, {scaffold['keyscale']}, {scaffold['timesignature']}/4")

    vocal_mode = random.random() < vocal_prob

    base_request = {
        "caption": caption,
        "duration": duration,
        "bpm": scaffold["bpm"],
        "keyscale": scaffold["keyscale"],
        "timesignature": scaffold["timesignature"],
        "lm_cfg_scale": _LM_CFG_SCALE,
        "lm_negative_prompt": _LM_NEGATIVE_PROMPT_VOCAL if vocal_mode else _LM_NEGATIVE_PROMPT,
        "use_cot_caption": False,
        # Turbo preset (SFT sounds worse despite 50 steps).
        "inference_steps": 8,
        "guidance_scale": 1.0,
        "shift": 3.0,
        "synth_model": ace_cfg.synth_model,
        "output_format": "mp3",
    }

    lyrics_source = "none"
    sampled_lyrics: str | None = None
    created_lyrics: str | None = None
    lyricist_seconds = 0.0
    if vocal_mode:
        # Rewrite caption to remove 'instrumental' wording so caption and
        # vocal_language don't pull the model in opposite directions. The
        # Zappa-style "free jazz" / "sound mass" textures are kept.
        vocal_caption = _vocalize_caption(caption, vocal_lang)
        if vocal_caption != caption:
            log(f"caption rewritten for vocal cycle: {vocal_caption[:160]}{'...' if len(vocal_caption) > 160 else ''}")
        base_request["caption"] = vocal_caption
        base_request["vocal_language"] = vocal_lang
        lyrics_transitions = sum(len(v) for v in lyrics_chain.values())
        want_markov_lyrics = (
            force_lyrics_mode == "markov"
            or (force_lyrics_mode == "auto" and lyrics_transitions >= _LYRICS_READY_THRESHOLD)
        )
        # Lyricist is a VOCAB FEEDER only — it generates a short fragment of
        # weird syllables/imagery and trains the markov lyrics_chain on it,
        # growing the chain's vocabulary cycle by cycle. The actual lyrics
        # sent to ACE come from sampling the chain — markov's "wrong word
        # on the rhythm" mismatch is what makes the result musical instead
        # of generic AI-pop. Clean lyricist output → clean AI music; chaotic
        # markov sample → chaotic interesting music.
        llama_fragment, lyricist_seconds = lyricist.generate_lyrics(
            vocal_caption,
            duration,
            bpm=scaffold["bpm"],
            description=caption,
            max_tokens=400,
            log=log,
        )
        if llama_fragment:
            added = train_lyrics(lyrics_chain, llama_fragment, lyrics_order)
            log(f"lyricist→vocab: +{added} chain transitions from {len(llama_fragment)} chars in {lyricist_seconds}s")

        # Sample from the (now-grown) chain for the actual lyrics.
        min_t, max_t = _lyric_token_bounds(duration)
        sampled_lyrics = sample_lyrics(
            lyrics_chain, lyrics_order,
            min_tokens=min_t, max_tokens=max_t,
        )
        if sampled_lyrics:
            created_lyrics = sampled_lyrics
            base_request["lyrics"] = sampled_lyrics
            lyrics_source = "markov"
            log(f"vocal cycle ({vocal_lang}); markov chain → {len(sampled_lyrics)} chars (target {min_t}-{max_t}t)")
        elif llama_fragment:
            # Chain too sparse to produce a usable sample yet — use the
            # lyricist fragment directly this cycle. Next cycle the chain
            # will have grown from this training pass.
            base_request["lyrics"] = llama_fragment
            lyrics_source = "llama3-bootstrap"
            log(f"vocal cycle ({vocal_lang}); chain too thin, using lyricist fragment directly ({len(llama_fragment)} chars)")
        else:
            lyrics_source = "lm"
            base_request["lyrics"] = ""
            log(f"vocal cycle ({vocal_lang}); lyricist + chain both empty — /lm will write lyrics")
    else:
        base_request["lyrics"] = "[Instrumental]"

    # Cover-mode shortcut for continuing takes of the same song. The composer
    # already played take 1; takes 2..N revisit it. ACE's task_type=cover
    # takes the previous take's mp3 as src_audio, VAE-encodes it, and renders
    # a new version influenced by both that source and the caption. The
    # transcribed lyrics from the previous take become this take's lyrics so
    # the composition's words "drift through re-transcription" each round.
    use_cover_mode = bool(
        song_state
        and not is_new_song
        and song_state.get("last_mp3_path")
        and Path(song_state["last_mp3_path"]).exists()
    )
    codes_source = "lm"
    enriched: dict | None = None
    lm_seconds = 0.0
    src_audio_for_synth: Path | None = None

    if use_cover_mode:
        prev_mp3 = Path(song_state["last_mp3_path"])
        prev_lyrics = (song_state.get("last_transcribed_lyrics") or "").strip()
        if prev_lyrics and prev_lyrics != "[Instrumental]":
            base_request["lyrics"] = prev_lyrics
            lyrics_source = "carryover"
            log(f"cover take: reusing {len(prev_lyrics)}-char transcribed lyrics from take {song_state.get('take_no') - 1}")
        base_request["task_type"] = "cover"
        base_request["audio_cover_strength"] = song_state.get("cover_strength", 0.6)
        # Cover mode skips the LM entirely (per ARCHITECTURE.md). We pass the
        # current base_request directly to /synth and the DiT does the work.
        enriched = base_request
        src_audio_for_synth = prev_mp3
        codes_source = "cover"
        log(f"cover mode: src={prev_mp3.name} strength={base_request['audio_cover_strength']}")
    else:
        codes_transitions = sum(len(v) for v in codes_chain.values())
        use_markov_codes = (
            force_codes_mode == "markov"
            or (force_codes_mode == "auto" and codes_transitions >= _CODES_READY_THRESHOLD)
        )
        if force_codes_mode == "lm":
            use_markov_codes = False
        # Vocal cycles must go through /lm: /synth decodes audio_codes directly
        # and ignores the lyrics field, so markov-sampled codes (trained on a
        # mix of vocal and instrumental audio) reproduce whatever was on the
        # tape, not what the lyrics ask for. /lm generates codes shaped by
        # caption + lyrics + vocal_language together.
        if vocal_mode and use_markov_codes:
            log("vocal cycle: routing through /lm so codes match the requested vocals")
            use_markov_codes = False

        if use_markov_codes:
            if not codes_chain:
                log("markov codes requested but chain is empty; falling back to /lm")
                use_markov_codes = False
            else:
                target_len = max(1, int(duration * 5))
                sampled, dead_ends = sample_codes(codes_chain, codes_order, target_len)
                log(f"sampled {len(sampled)} markov codes ({dead_ends} dead-end jump(s))")
                base_request["audio_codes"] = format_codes(sampled)
                base_request["vocal_language"] = "en"
                enriched = base_request
                codes_source = "markov"

        if not use_markov_codes:
            t0 = time.time()
            log(f"SENDING TO /LM: {json.dumps(base_request, indent=2)}")
            enriched = ace_client.lm(ace_cfg, base_request, log=log)
            t1 = time.time()
            lm_seconds = round(t1 - t0, 2)
            log(f"/lm done in {t1 - t0:.1f}s (bpm={enriched.get('bpm')} key={enriched.get('keyscale')})")
            if vocal_mode:
                generated_lyrics = enriched.get("lyrics") or ""
                log(f"LM generated lyrics: {generated_lyrics[:100]!r}{'...' if len(generated_lyrics) > 100 else ''}")

    t2 = time.time()
    mp3, latent = ace_client.synth(ace_cfg, enriched, src_audio=src_audio_for_synth, log=log)
    t3 = time.time()
    log(f"/synth done in {t3 - t2:.1f}s (mp3={len(mp3)} bytes, codes={codes_source})")

    _ensure_audio_dir(paths.audio_dir)
    base = f"{slot:02d}"
    mp3_path = paths.audio_dir / f"{base}.mp3"
    json_path = paths.audio_dir / f"{base}.json"
    mp3_path.write_bytes(mp3)
    # Persist the freshly-written mp3 path for the *next* take of this song
    # to consume as src_audio in cover mode.
    if song_state is not None:
        song_state["last_mp3_path"] = str(mp3_path)
    # Log the lyrics that were actually fed to /synth this cycle. Mirrors
    # state/understood_lyrics.txt so the two can be diffed: what we asked
    # ACE to sing vs what the transcriber heard come back.
    append_used_lyrics(
        paths,
        cycle=cycle,
        mp3_name=mp3_path.name,
        enriched=enriched,
        duration=enriched.get("duration"),
        vocal_mode=vocal_mode,
        lyrics_source=lyrics_source,
    )

    sidecar = {
        "cycle": cycle,
        "slot": slot,
        "caption": caption,
        "enriched_caption": enriched.get("caption"),
        "scaffold": scaffold,
        "codes_source": codes_source,
        "vocal_mode": vocal_mode,
        "lyrics_source": lyrics_source,
        "created_lyrics": created_lyrics,
        "used_lyrics": enriched.get("lyrics"),
        "bpm": enriched.get("bpm"),
        "keyscale": enriched.get("keyscale"),
        "timesignature": enriched.get("timesignature"),
        "vocal_language": enriched.get("vocal_language"),
        "duration": enriched.get("duration"),
        "seed": enriched.get("seed"),
        "lm_seed": enriched.get("lm_seed"),
        "lyrics": enriched.get("lyrics"),
        "lm_model": enriched.get("lm_model"),
        "synth_model": enriched.get("synth_model"),
        "inference_steps": enriched.get("inference_steps"),
        "lm_seconds": lm_seconds,
        "lyricist_seconds": lyricist_seconds,
        "synth_seconds": round(t3 - t2, 2),
        "audio_codes": enriched.get("audio_codes"),
        "trained_caption": None,
        "code_count": None,
    }

    # If we went through /lm, those audio_codes are ACE's idea of this caption.
    # Feed them back into the codes chain — that's the cheapest seed source.
    if codes_source == "lm":
        seq = parse_codes(enriched.get("audio_codes") or "")
        if seq:
            added = train_codes(codes_chain, seq, codes_order)
            sidecar["code_count"] = len(seq)
            log(f"trained codes chain (+{added} from /lm; len={len(seq)})")
    # Lyric feedback:
    # - LM-authored lyrics seed the chain directly.
    # - Markov-authored lyrics that /lm format changed replace the raw sampled
    #   draft in the chain, so the lyric memory evolves toward the processed
    #   version rather than accumulating both forms.
    lm_lyrics = enriched.get("lyrics") or ""
    if vocal_mode and lm_lyrics and lm_lyrics.strip() != "[Instrumental]":
        if lyrics_source == "lm":
            added = train_lyrics(lyrics_chain, lm_lyrics, lyrics_order)
            log(f"trained lyrics chain (+{added} from /lm)")
        elif lyrics_source == "markov+format" and created_lyrics and lm_lyrics != created_lyrics:
            removed = untrain_lyrics(lyrics_chain, created_lyrics, lyrics_order)
            added = train_lyrics(lyrics_chain, lm_lyrics, lyrics_order)
            log(f"replaced created lyrics in chain (-{removed} raw, +{added} /lm-formatted)")
    json_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n")
    log(f"wrote {mp3_path.name}")

    if autofeedback:
        try:
            t4 = time.time()
            meta, _latent = ace_client.understand(ace_cfg, mp3_path, log=log)
            t5 = time.time()
            log(f"/understand done in {t5 - t4:.1f}s")
            trained = metadata_caption(meta)
            sidecar["understand_meta"] = {
                k: meta.get(k) for k in
                ("caption", "bpm", "keyscale", "timesignature", "vocal_language", "duration")
            }
            sidecar["understand_seconds"] = round(t5 - t4, 2)

            # Train the codes chain on what /understand actually heard.
            seq = parse_codes(meta.get("audio_codes") or "")
            if seq:
                added = train_codes(codes_chain, seq, codes_order)
                log(f"trained codes chain (+{added} from /understand; len={len(seq)})")
            # The ACE-Step transcriber (Qwen2.5-Omni FP8) on vLLM gives
            # better lyrics than ace-understand for our own synthetic audio.
            # Try it first; fall back to /understand's lyrics on any failure
            # so the loop keeps running if vLLM is down.
            transcriber_lyrics = ""
            transcriber_lang = ""
            try:
                if cli_transcribe.server_alive(cli_transcribe.DEFAULT_BASE_URL):
                    t_t0 = time.time()
                    t_result = cli_transcribe.transcribe_one(
                        cli_transcribe.DEFAULT_BASE_URL,
                        cli_transcribe.DEFAULT_MODEL,
                        mp3_path,
                    )
                    raw_lyrics = (t_result["parsed"].get("lyrics") or "").strip()
                    transcriber_lang = (t_result["parsed"].get("languages") or "").strip()
                    sidecar["transcriber_lang"] = transcriber_lang
                    sidecar["transcriber_seconds"] = round(time.time() - t_t0, 2)
                    if t_result.get("runaway"):
                        # Templated loop ("[Melodic Theme N: Violin]" cycling
                        # by digit) — looks like content but contributes
                        # nothing musical. Keep on disk for inspection but
                        # don't use, train, or carry over.
                        sidecar["transcriber_lyrics_raw"] = raw_lyrics
                        sidecar["transcriber_runaway"] = t_result.get("runaway_reason")
                        log(f"transcriber: REJECTED ({t_result.get('runaway_reason')})")
                    else:
                        transcriber_lyrics = raw_lyrics
                        sidecar["transcriber_lyrics"] = transcriber_lyrics
                        log(f"transcriber: {len(transcriber_lyrics)} chars, lang={transcriber_lang!r}")
            except Exception as exc:  # noqa: BLE001
                log(f"transcriber failed (continuing with /understand lyrics): {exc!r}")

            # Stash for the next take of this song: cover mode will feed
            # mp3_path back as src_audio and the transcribed lyrics back
            # as the lyrics field, so each take builds on the last.
            if song_state is not None and song_state.get("takes_left", 0) > 0:
                song_state["last_transcribed_lyrics"] = (
                    transcriber_lyrics or (meta.get("lyrics") or "")
                )
                song_state["last_audio_codes"] = meta.get("audio_codes") or ""

            # Prefer transcriber lyrics over /understand's lyrics for training
            # and corpus-building. /understand still drives codes + caption.
            heard_lyrics = transcriber_lyrics or (meta.get("lyrics") or "")
            if vocal_mode and heard_lyrics and heard_lyrics.strip() != "[Instrumental]":
                added = train_lyrics(lyrics_chain, heard_lyrics, lyrics_order)
                log(f"trained lyrics chain (+{added} from {'transcriber' if transcriber_lyrics else '/understand'})")
            # Inject the chosen lyrics into a meta copy so the corpus header
            # reflects what was actually trained on.
            corpus_meta = dict(meta)
            corpus_meta["lyrics"] = heard_lyrics
            if transcriber_lang:
                corpus_meta["vocal_language"] = transcriber_lang
            append_understood_lyrics(
                paths,
                cycle=cycle,
                mp3_name=mp3_path.name,
                meta=corpus_meta,
                duration=enriched.get("duration"),
                vocal_mode=vocal_mode,
                lyrics_source=f"transcriber+{lyrics_source}" if transcriber_lyrics else lyrics_source,
            )

            if not trained:
                log("understand returned no usable caption; skipping text train")
            else:
                added = train_text(chain, trained, order)
                save_chain(chain, order, paths.chain_path)
                sidecar["trained_caption"] = trained
                log(f"trained text chain (+{added}): {trained[:160]}{'...' if len(trained) > 160 else ''}")
            json_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001 — keep the loop alive
            log(f"autofeedback failed: {exc!r}")
    # Persist chains after each cycle (cheap; survives crashes).
    save_codes_chain(codes_chain, codes_order, paths.codes_chain_path)
    save_lyrics_chain(lyrics_chain, lyrics_order, paths.lyrics_chain_path)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="markovsound", description="generative music loop")
    parser.add_argument("-d", "--duration", type=int, default=_DEFAULT_DURATION,
                        help="fixed track duration in seconds; 0 = use live state/runtime.json duration range")
    parser.add_argument("-n", "--cycles", type=int, default=0,
                        help="stop after N cycles (0 = run forever)")
    parser.add_argument("--no-autofeedback", action="store_true",
                        help="skip /understand training step")
    parser.add_argument("--no-autostart", action="store_true",
                        help="do not auto-start ace-server")
    parser.add_argument("--order", type=int, default=_DEFAULT_ORDER,
                        help="markov order when building from corpus")
    parser.add_argument("--codes-mode", choices=["auto", "markov", "lm"], default="auto",
                        help="audio_codes source: 'auto' = markov once chain has enough, "
                             "'markov' = always force markov (errors if empty), 'lm' = always /lm")
    parser.add_argument("--vocal-lang", default="en",
                        help="ISO 639-1 vocal language (en, fr, de, nl, es, it, ja, ...)")
    parser.add_argument("--lyrics-mode", choices=["auto", "markov", "lm"], default="auto",
                        help="lyrics source on vocal cycles: 'auto' = markov once chain ready, "
                             "with /lm fallback if no sample contains a vocal section; "
                             "'markov' = always force chain (errors if empty); 'lm' = always /lm")
    parser.add_argument("--no-wait-for-player", action="store_true",
                        help="don't pause between cycles when ./play is running — "
                             "go full speed so the chain evolves faster")
    args = parser.parse_args(argv)

    paths = Paths.discover()
    ace_cfg = AceConfig.discover()
    log(f"paths: state={paths.state_dir} audio={paths.audio_dir}")
    log(f"ace: {ace_cfg.base_url} models={ace_cfg.models_dir}")

    chain, order = _load_or_build_chain(paths, args.order)
    codes_chain, codes_order = _load_codes_chain(paths)
    lyrics_chain, lyrics_order = _load_lyrics_chain(paths)

    stop_flag = {"requested": False}

    def _handle_sigint(signum, frame):  # noqa: ARG001
        # async-signal-safe: write directly, no logger lock
        os.write(2, b"\n[signal] SIGINT received; will stop after current cycle.\n")
        if stop_flag["requested"]:
            os.write(2, b"[signal] second SIGINT; exiting immediately.\n")
            os._exit(130)
        stop_flag["requested"] = True

    signal.signal(signal.SIGINT, _handle_sigint)

    proc = None
    if not args.no_autostart:
        proc = ace_client.ensure_server(ace_cfg, log=log)
    elif not ace_client.server_alive(ace_cfg):
        log(f"ace-server not running at {ace_cfg.base_url} and --no-autostart given; exiting.")
        return 1

    try:
        cycle = _read_cycle(paths)
        completed = 0
        try:
            runtime_cfg = load_runtime_config(paths.runtime_config_path)
        except ValueError as exc:
            runtime_cfg = RuntimeConfig()
            log(f"runtime config invalid at startup; using defaults: {exc}")
        log(f"runtime config: {runtime_cfg}")
        # Persistent song state — survives across cycles within one ./create
        # invocation so multiple takes share caption+slot+duration. Cleared
        # automatically when takes_left hits 0.
        song_state: dict = {"caption": None, "slot": None, "duration": 0,
                            "takes_left": 0, "take_no": 0,
                            "cover_strength": args.cover_strength,
                            "last_mp3_path": None,
                            "last_transcribed_lyrics": "",
                            "last_audio_codes": "",
                            "runtime_cfg": runtime_cfg}
        while True:
            cycle += 1
            try:
                try:
                    runtime_cfg = load_runtime_config(paths.runtime_config_path)
                except ValueError as exc:
                    log(f"runtime config invalid; keeping previous values: {exc}")
                else:
                    if runtime_cfg != song_state.get("runtime_cfg"):
                        log(f"runtime config: {runtime_cfg}")
                        song_state["runtime_cfg"] = runtime_cfg
                    song_state["cover_strength"] = runtime_cfg.cover_strength
                ok = _run_cycle(
                    cycle, chain, order, codes_chain, codes_order,
                    lyrics_chain, lyrics_order, paths, ace_cfg, runtime_cfg,
                    duration=args.duration,
                    autofeedback=not args.no_autofeedback,
                    force_codes_mode=args.codes_mode,
                    vocal_prob=runtime_cfg.vocal_prob,
                    vocal_lang=args.vocal_lang,
                    force_lyrics_mode=args.lyrics_mode,
                    song_state=song_state,
                    takes_per_song=runtime_cfg.takes_per_song,
                )
            except ace_client.AceError as exc:
                log(f"cycle {cycle} failed: {exc}")
                ok = False
            if ok:
                completed += 1
                _write_cycle(paths, cycle)
            if stop_flag["requested"]:
                log("stop requested; exiting loop.")
                break
            if args.cycles and completed >= args.cycles:
                log(f"completed {completed} cycle(s); exiting.")
                break
            # Back-pressure: if ./play is running, wait for the current track
            # to finish before generating the next one so each generated slot
            # actually gets played. With no player, this returns immediately
            # and create runs at full speed. --no-wait-for-player skips this
            # entirely so the chain evolves as fast as ace-server can synth.
            if not args.no_wait_for_player:
                wait_for_player(paths, stop_flag=stop_flag, log=log)
        return 0
    finally:
        ace_client.shutdown_server(proc, log=log)


if __name__ == "__main__":
    sys.exit(main())
