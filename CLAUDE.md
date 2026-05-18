# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MarkovSound is a generative *music* system in the same spirit as its sibling
project [MarkovArt](../MarkovArt) (see `MARKOVART_REFERENCE.md` for the
inherited architecture). It combines Markov chain text generation with the
ACE-Step diffusion music model to produce audio from procedurally evolved
captions.

Pipeline (mirrors MarkovArt, audio instead of image):

```
state/markov_chain.pkl → caption (+ lyrics) → ace-server → Audio/*.mp3
        ↑                                                      |
        └── autofeedback: ace-understand tags audio → train caption ──┘
```

The image-feedback loop in MarkovArt uses a VLM (Gemma via `llama-mtmd-cli`)
to describe generated images. MarkovSound uses **`ace-understand`** — the
reverse pipeline shipped with acestep.cpp — which takes an MP3/WAV and
returns lyrics + metadata (caption, BPM, key, language, time signature).
That metadata becomes the training text fed back into the chain.

## External dependency: acestep.cpp

The diffusion backend lives at `~/Src/acestep.cpp/` and is not vendored
here. Two entry points matter:

- **`ace-server`** — HTTP server on port 8085 (default). Endpoints:
  - `POST /lm` — caption → lyrics + audio codes (JSON)
  - `POST /synth` — codes → MP3/WAV (multipart: audio + latent)
  - `POST /understand` — audio → metadata + lyrics + codes (multipart)
  - `POST /vae` — encode/decode latents
  - `GET /health`, `GET /props`
- **`ace-lm` + `ace-synth` + `ace-understand`** — standalone CLIs that take
  a JSON request and pipe through the same stages. Prefer these for batch
  scripts; prefer the server for interactive/live loops.

Required GGUF model files in `~/Src/acestep.cpp/models/`:

| Type | Default |
|------|---------|
| LM | `acestep-5Hz-lm-4B-Q8_0.gguf` |
| Text encoder | `Qwen3-Embedding-0.6B-Q8_0.gguf` |
| DiT | `acestep-v15-turbo-Q8_0.gguf` (8-step turbo) or `…-sft-Q8_0.gguf` (50-step quality) |
| VAE | `vae-BF16.gguf` |

### AceRequest JSON shape (essentials)

```json
{
  "lm_model": "acestep-5Hz-lm-4B-Q8_0.gguf",
  "synth_model": "acestep-v15-turbo-Q8_0.gguf",
  "caption": "<the Markov-generated prompt>",
  "lyrics": "<optional; LM will write them if omitted>",
  "vocal_language": "en|nl|fr|…",
  "duration": 120,
  "bpm": 124,
  "keyscale": "F# major",
  "timesignature": "4",
  "inference_steps": 8,
  "guidance_scale": 1.0,
  "shift": 3.0
}
```

Full reference: `~/Src/acestep.cpp/docs/ARCHITECTURE.md`.

## Translation table (MarkovArt → MarkovSound)

| MarkovArt | MarkovSound |
|-----------|-------------|
| Flux / Z-Image (sd-cli) | ACE-Step (ace-server / ace-synth) |
| VLM description (`llama-mtmd-cli`) | `ace-understand` metadata + lyrics |
| `prompts/describe_image.txt` | `prompts/describe_audio.txt` (caption template for re-training) |
| `prompts/viewpoint.md` | `prompts/listening_stance.md` |
| `Images/` + `Images/absorb/` | `Audio/` + `Audio/absorb/` |
| `state/painters/*.md` (possession) | `state/composers/*.md` (possession) |
| `./fetch "oxidized copper"` | `./fetch "Detroit techno 1995"` (reference tracks → `Audio/absorb/`) |
| `--aspect 5x4` | `--duration 90` (and/or `--bpm`, `--keyscale`) |
| Jaccard dedup on prompts | Jaccard dedup on captions (+ optional acoustic dedup later) |
| `state/identity.md` | `state/identity.md` (artistic identity for *sonic* output) |

## Common commands (planned, mirroring MarkovArt)

```bash
./create                       # main generation loop
./create -d 60 --bpm 90        # 60s tracks at 90 BPM
./fetch "ambient kosmische"    # download reference audio to Audio/absorb/
./trigger                      # force evolve + mutate next cycle
./words list / block / replace # vocabulary management
./verwijder <track.mp3>        # remove track and untrain its caption
```

## Directory structure (target, mirrors MarkovArt)

```
MarkovSound/
  create               # main generation loop command
  fetch                # fetch reference audio (yt-dlp / freesound / …)
  trigger              # manually trigger evolve/mutate cycle
  verwijder            # remove track and untrain its caption
  words                # word overrides and replacements
  pyproject.toml
  prompts/             # describe_audio.txt, listening_stance.md
  src/markovsound/     # Python package (core logic)
  scripts/             # CLI utilities
  state/               # runtime state (chain, identity, memory, …)
    markov_chain.pkl
    prompt_corpus.txt
    identity.md
    composers/         # possession personas (was: painters/)
    subconscious.jsonl # optional memory echoes
  Audio/               # generated tracks + absorb/ for references
  website/             # live player (now playing + chat panel)
  attic/               # archived experiments
```

## Possession (composers)

Same loader rules as MarkovArt's painters:

- Non-recursive `glob("*.md")` in `state/composers/`
- Skips `_`-prefixed files and `COMPOSERFILE.md`
- YAML frontmatter: `name`, `weight` (explicit weights sum ≤ 0.90,
  remainder reserved for "pure mode")
- Subdir `_archive/` for retired personas

Composers should be invented (à la MarkovArt's post-reset painters), not a
canonical Top-40 list. The lens each persona provides — material, technique,
constraint, ritual — matters more than the name.

## Differences from MarkovArt to design through

1. **Captions vs. prompts.** ACE-Step captions are denser and more
   structured than diffusion prompts (instruments, mood, BPM, key,
   language). The Markov chain may need higher order (3?) and/or a
   structured-section sampler.
2. **Lyrics.** Optional but powerful. Either let the LM auto-write them
   (omit `lyrics`) or evolve a second chain trained on lyric corpora.
3. **Long generation time.** A 120s track at 8 steps is still slower than
   a single image. Cycle budgets and `target-images` analogues need
   re-tuning. Plan for fewer-but-longer artifacts.
4. **Autofeedback granularity.** `ace-understand` gives BPM/key/language
   *and* a lyrics transcription. Train the chain on the metadata caption,
   *not* the lyrics (lyrics live in a separate corpus).
5. **Storage.** MP3s at 128 kbps are ~1 MB/min — manageable. Latents
   (`.vae` files) are larger but enable repaint/cover modes without
   re-encoding.

## Status

Bootstrap only. No `src/`, no `state/`, no scripts yet. Start by porting
the MarkovArt skeleton (`loop.py`, `markov.py`, `evolve.py`,
`possession.py`, `curation.py`, `config.py`) and swapping the inference +
description backends.

## No test or lint framework

Same as MarkovArt: no unit tests, no linter, no CI. Verify by running and
listening.
