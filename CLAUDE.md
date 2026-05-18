# CLAUDE.md

This file gives coding-oriented context for the current MarkovSound system.
For operator-facing commands, start with `USER_GUIDE.md`. For the conceptual
learning model, see `HOW_MARKOVSOUND_IMPROVES.md`. `MARKOVART_REFERENCE.md`
is historical reference material from the sibling project, not a live spec.

## What MarkovSound is

MarkovSound is a generative music loop built around ACE-Step. It composes from
three evolving Markov memories, renders tracks through `ace-server`, listens
back to its own output, and folds selected observations into future output.

```text
text / lyrics / audio-code chains
            ↓
      plan one cycle
            ↓
       ACE-Step synth
            ↓
       Audio/staging/
            ↓
 understand + transcribe + feedback
            ↓
        Audio/queue/ → ./play → Audio/archive/
```

## Runtime flow

`./create` runs `markovsound.loop`.

1. `loop.py` handles lifecycle, signals, live config reloads, chain hot-reload,
   and buffer scheduling.
2. `cycle.py` runs one cycle through explicit phases:
   - `plan_song`
   - `prepare_lyrics`
   - `synthesize`
   - `write_output`
   - `train_generated`
   - `apply_feedback`
3. Complete tracks are staged first, then published to `Audio/queue/` only
   after feedback is finished.
4. `./play` consumes the oldest queued item, exposes `INS`/`DEL` curation, and
   archives finished tracks.

## The three memories

| Chain | File | Role |
|---|---|---|
| Text | `state/markov_chain.pkl` | Generates descriptive captions / song concepts. |
| Lyrics | `state/lyrics_chain.pkl` | Generates vocal structures, words, and stage directions. |
| Codes | `state/codes_chain.pkl` | Generates learned ACE audio-code sequences when the loop uses Markov-code mode. |

The text chain deliberately trains on descriptive prose only. BPM, key, meter,
and vocal-language metadata remain in JSON sidecars; they are not appended to
caption training text because they degrade into fragments such as
`bpm, in d minor, time`.

## Live state and configuration

Working state is intentionally outside source control:

- `Audio/staging/` — in-progress outputs not yet visible to the player
- `Audio/queue/` — complete unplayed tracks
- `Audio/archive/` — finished playback history
- `Audio/absorb/` — reference tracks for training
- `state/runtime.json` — live editable loop settings
- `state/*_chain.pkl` — serialized Markov chains
- `state/used_lyrics.txt` / `state/understood_lyrics.txt` — lyric audit corpora

`state/runtime.json` is reloaded before every cycle. The chain pickle files are
also hot-reloaded between cycles when maintenance tools modify them externally.
That means commands such as `./replace-lyrics` can be used while `./create` is
running.

Named sessions isolate complete working worlds under `sessions/NAME/`. The root
`state/` and `Audio/` paths are symlinks to the active session, selected by
`.markovsound_session` or a one-command `MARKOVSOUND_SESSION=NAME` override.
`./session` manages creation, save/clone, activation, rename, and deletion.
The loop checks for session changes at cycle boundaries and reloads chains,
config, song carry-over, and cycle count when one occurs.

## Main modules

| Module | Responsibility |
|---|---|
| `loop.py` | Process orchestration, config reloads, queue back-pressure, signal handling. |
| `cycle.py` | One complete generation cycle and its feedback phases. |
| `song.py` | Prompt shaping and per-take musical scaffold helpers. |
| `markov.py` | Caption-chain construction, sampling, training, cleanup. |
| `lyrics_markov.py` | Structured lyric tokenization and lyric-chain logic. |
| `codes_markov.py` | Audio-code chain logic. |
| `ace_client.py` | ACE server lifecycle and HTTP job calls. |
| `runtime_config.py` | Validation and persistence for live runtime settings. |
| `playback.py` | Queue / player coordination. |
| `describe.py` | Feedback-caption normalization and filtering. |

## External dependencies

ACE-Step lives outside this repository, usually at `~/Src/acestep.cpp/`.
Defaults assume:

- server: `build/ace-server`
- host/port: `127.0.0.1:8085`
- models: `models/`
- LM: `acestep-5Hz-lm-4B-Q8_0.gguf`
- synth model: `acestep-v15-turbo-Q8_0.gguf`

`ace-server` endpoints used here:

- `POST /lm`
- `POST /synth`
- `POST /understand`
- `GET /health`

The optional external transcriber used by `cli_transcribe.py` is separate from
`ace-understand` and may be unavailable; the loop degrades gracefully.

## Common development commands

```bash
python3 -m pip install -e ".[dev]"
PYTHONPATH=src pytest -q
./create
./play
./absorb Audio/absorb
./steer "prepared piano drones"
./replace-lyrics Losers Dancers --dry-run
./clean-caption-metadata --dry-run
./session current
./session save experiment-a --use
python3 scripts/smoke_ace.py
python3 scripts/smoke_vocals.py
```

Use smoke scripts for ACE-dependent verification; pure logic belongs in
`tests/`.
