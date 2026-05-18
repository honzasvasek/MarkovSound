# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `src/markovsound/`. The main loop is in `loop.py`; focused modules such as `cycle.py`, `markov.py`, `lyrics_markov.py`, `codes_markov.py`, and `ace_client.py` hold generation and backend logic. Root wrappers (`create`, `play`, `absorb`, `keep`, `steer`, `verwijder`) provide the operator interface. Operator-facing Python CLIs live in `src/markovsound/` and are also exposed as installable `markovsound-*` console scripts. `scripts/` is reserved for thin compatibility wrappers, smoke tests, and non-Python helpers.

`Audio/` contains queued tracks in `Audio/queue/`, in-progress generation outputs in `Audio/staging/`, archived tracks, metadata sidecars, reference material in `Audio/absorb/`, and smoke-test outputs. `state/` contains mutable runtime data such as pickle chains, corpora, steering settings, and live `runtime.json` loop settings. Treat both directories as working data, not source code. Project notes live in `USER_GUIDE.md`, `HOW_MARKOVSOUND_IMPROVES.md`, `CLAUDE.md`, and `MARKOVART_REFERENCE.md`.

## Build, Test, and Development Commands

- `python3 -m pip install -e .` — install the package in editable mode.
- `./create` — run the live generation loop through `markovsound.loop`.
- `./play` — listen to queued tracks and use `INS`/`DEL` curation controls.
- `./absorb Audio/absorb` — analyze reference audio and train the chains.
- `./steer "prepared piano drones"` — bias the text chain toward a target style.
- `PYTHONPATH=src python3 -m markovsound.cli_dedup` — flatten chain weights when output becomes repetitive.
- `python3 scripts/smoke_ace.py` — exercise ACE-Step generation end to end.
- `python3 scripts/smoke_vocals.py` — run a vocal-generation smoke test against a live server.

ACE-Step is external; local generation assumes a reachable `ace-server` and model files as described in `CLAUDE.md`.

## Coding Style & Naming Conventions

Use Python 3.11+, four-space indentation, type hints where practical, and `snake_case` for modules, functions, and variables. Existing files prefer small single-purpose modules, `argparse` CLIs, `pathlib.Path`, and `from __future__ import annotations`; follow those patterns. Shell wrappers should stay minimal and delegate real logic to package modules.

## Testing Guidelines

Pure logic is covered with `pytest` tests under `tests/`; run them with `PYTHONPATH=src pytest -q`. Keep ACE-server-dependent behavior in smoke scripts, and for loop changes still do a short `./create` run plus inspection of generated `.mp3`/`.json` pairs when the backend is available.

## Commit & Pull Request Guidelines

This checkout does not include Git history, so no repository-specific commit convention can be inferred. Use concise imperative commits such as `Add lyric chain deduplication`. Pull requests should explain the behavioral change, list manual verification performed, call out any `state/` migration implications, and include sample output or logs when audio-generation behavior changes.

## Runtime Data Safety

Avoid committing incidental updates from `Audio/` or `state/` unless they are intentional fixtures or seed data. Retraining, deletion, and deduplication alter future output; document those effects clearly.
