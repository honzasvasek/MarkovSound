# MarkovArt — Reference Structure

This document captures the structure of the sibling project
[MarkovArt](../MarkovArt) at the moment MarkovSound was forked from its
ideas. It is a *reference snapshot*, not a live spec — MarkovArt continues
to evolve independently.

When in doubt about a design decision, look at the equivalent file in
MarkovArt first. The intent is for MarkovSound to mirror MarkovArt's
shape so that bug fixes, refactors, and conceptual improvements can flow
between the two projects.

## Pipeline

```
state/markov_chain.pkl  →  prompts  →  sd-cli  →  Images/*.png
        ↑                                     |
        └──── autofeedback: VLM describes output → train description ─────┘
```

Optional: `state/subconscious.jsonl` feeds semantically related memory
echoes back into the prompt-upgrade step.

## Directory layout

```
MarkovArt/
  create              # main generation loop command
  fetch               # fetch reference images from the web
  trigger             # manually trigger evolve/mutate cycle
  verwijder           # remove words/phrases from the chain
  words               # manage word overrides and replacements
  pyproject.toml      # installable Python package config
  prompts/            # describe_image.txt, viewpoint.md
  src/markovart/      # Python package (core logic)
  scripts/            # wrappers, smoke tests, and non-Python helpers
  state/              # runtime state (chain, identity, memory, …)
  Images/             # generated images + absorb/
  website/            # live viewer (crossfade + chat panel)
  Playground/         # experiments and one-off scripts
  attic/              # old versions and archived files
```

## Key Python modules (in `src/markovart/`)

| Module | Responsibility |
|--------|----------------|
| `loop.py` | Main generation loop, CLI entry point, signal handling, live config polling, chat server |
| `markov.py` | Order-2 Markov chain: build, sample, train, untrain, absorb |
| `evolve.py` | Autonomous LLM-driven evolution and mutation (Claude / Gemini / Codex) |
| `describe.py` | VLM image description via `llama-mtmd-cli` (Gemma 4 26B) |
| `identity.py` | Artistic identity bootstrap and refresh |
| `curation.py` | Image pool curation with Jaccard deduplication |
| `possession.py` | Painter persona selection + prompt-block injection |
| `subconscious.py` | Memory echoes from prior prompt/description pairs |
| `metadata.py` | PNG metadata read/write via `exiftool` |
| `config.py` | Path and model discovery |
| `live_config.py` | Polls `state/loop.json` each cycle for runtime parameter changes |
| `cli_session.py` | Session switching support |

## Key state files (in `state/`)

| File | Contents |
|------|----------|
| `prompt_corpus.txt` | Art-direction prompts; source of truth when rebuilding the chain |
| `markov_chain.pkl` | Serialized `{"chain": dict, "order": int}` (order-2) |
| `identity.md` | Current artistic identity document (symbiosis of user + Claude voice) |
| `identity.md.boot` | Fallback identity for cold-start |
| `subconscious.jsonl` | Optional memory log of prompt/description pairs |
| `evolution_memory.json` | History of evolve + mutation rationales, next-trigger cycles |
| `cycle.txt` | Current cycle counter |
| `loop.json` | Live-editable runtime configuration (14 keys) |
| `painters/*.md` | Possession personas — each with `name`, `weight` frontmatter |
| `painters/_archive/` | Retired personas (loader is non-recursive, so they're skipped) |
| `painters/PAINTERFILE.md` | Template / docs for the painter file format |
| `backups/` | Pre-reset snapshots |

## Markov chain internals

- Order-2: transition table keyed by word-pair tuples → `{next_word: weight}`
- Special tokens: `__START__` / `__END__` for sentence boundaries
- Temperature sampling (T>1 flattens, T<1 sharpens)
- `absorb_text()` injects new vocabulary with controlled ratio (0.0–1.0)
- `train_text()` / `untrain_text()` for fine-tuning from feedback

## Autonomous evolution

- `evolve.py` fires on a configurable cycle interval (50–500 cycles)
- `run_mutation()` fires independently on a configurable shorter interval
- Both call an LLM to steer the chain
- `./trigger` sets next-trigger cycle to 0 for immediate activation
- `./session use <name>` switches the running loop to a different session on the next cycle

## Possession (painters)

- Loader: non-recursive `glob("*.md")` in `state/painters/`
- Skips `_`-prefixed files and `PAINTERFILE.md`
- YAML frontmatter: `name` (display) and `weight` (probability)
- Explicit weights sum ≤ 0.90, minimum 10 % reserved for "pure mode" (no possession)
- Current set (post-2026-04 reboot): 10 invented painters with one-line conceptual lenses
  — e.g. Kestrel Vähämäki (blind-practice), Viveka Halász-Prenn (slime mold cartographer),
  Rhodopé Astrakhanian (forensic autobiography), Øde Ranganathan (threshold painter), …
- Archived: 14 historical painters in `_archive/` (Kandinsky, Beuys, Bourgeois,
  Hijikata, Hesse, Bacon, Mendieta, Klint, Anadol, Paik, Kusama, Matta-Clark,
  Schneemann, claude)

## Live configuration (`state/loop.json`)

Polled every cycle. 14 editable keys: `model`, `aspect`, `batch`,
`autofeedback`, `imagine`, `possession`, `possession_duration`,
`force_mutate`, `photo_ratio`, `random_curate`, `evolve`, `max_images`,
`target_images`, `prefill`.

## Web interface

Chat server on port 8082:

- `POST /chat` — chat with the system (actions surface for evolve, mutate, etc.)
- `GET /chat/state` — current state for the viewer
- `GET/POST /config` — read/write `loop.json` from the browser
- Viewer UI: crossfade slideshow + side chat panel

## Signal handling

- `os.write()` in SIGINT handler for async-signal safety (`log()` would deadlock
  the stdout mutex)
- `subprocess.DEVNULL` stdin isolation for child inference processes
- Second Ctrl-C forwards SIGINT to the active child `Popen` so it actually exits

## External dependencies

- `sd-cli` — diffusion inference (not in repo)
- `exiftool` — PNG metadata read/write
- `llama-mtmd-cli` — VLM inference for descriptions
- `llama-cli` — LLM inference for evolution
- Model files in `~/Models/`

## Common commands

```bash
./create                        # run generation loop (zimage, 5x4, autofeedback, evolve)
./create -m flux -a 16x9        # Flux model, 16:9 aspect
./create -b 3                   # 3 images per prompt
./fetch "oxidized copper"       # fetch reference images into Images/absorb/
./trigger                       # trigger evolve + mutate on next cycle
./trigger evolve                # trigger only evolve
./words list                    # list blocked/replaced words
./words block <word>            # block a word from generation
./words replace <old> <new>     # replace word in all prompts
./verwijder <image.png>         # remove image and untrain its prompt
```

---

When porting, the rough mapping is:

- `loop.py` → keep almost verbatim; swap the inference call and the
  description call.
- `markov.py`, `evolve.py`, `curation.py`, `possession.py`, `live_config.py`,
  `cli_session.py`, `subconscious.py` → port unchanged, rename module path.
- `describe.py` → rewrite around `ace-understand` (HTTP or CLI).
- `config.py` → point at acestep.cpp paths instead of `~/Models/`.
- `metadata.py` → MP3 tags via `mutagen` (or eyeD3) instead of `exiftool` PNG fields.
