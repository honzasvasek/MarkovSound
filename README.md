# MarkovSound

MarkovSound is an autonomous generative music system built around evolving
Markov memories and an external ACE-Step synthesis server. It composes tracks,
listens back to what it rendered, learns from the result, and lets a human
listener steer what survives.

The aim is not polished, interchangeable music. MarkovSound explores the
intersection of contemporary composition, free improvisation, unstable song
forms, machine listening, and productive digital artefacts.

```text
caption / lyric / audio-code memories
                  │
                  ▼
       concept + dramatic form
                  │
                  ▼
             ACE-Step render
                  │
                  ▼
       understand + transcribe
                  │
                  ▼
       learn, keep, delete, steer
                  │
                  └──────────────► next composition
```

## What makes it different

- **Three evolving memories:** descriptive captions, structured lyrics, and
  raw ACE audio-code trajectories develop independently.
- **Self-listening:** generated music can be analysed and fed back into the
  appropriate chains.
- **Human taste as pressure:** `keep` reinforces material; `verwijder` removes
  a track and untrains its contribution.
- **Dramatic form:** temporary arcs such as erosion, rupture, accumulation, and
  false return shape a track without polluting the learned caption vocabulary.
- **Related takes:** cover generations carry audio and heard lyrics forward,
  turning retries into a lineage of transformations.
- **Live steering:** runtime settings and chain maintenance can change between
  cycles without restarting the loop.
- **Isolated sessions:** each musical world keeps its own memories, audio,
  queue, archive, and configuration.

## Requirements

- Python 3.11 or newer
- An ACE-Step `ace-server` build with compatible `/lm`, `/synth`, and
  `/understand` endpoints
- ACE language and synthesis model files
- `mpv` for the interactive `./play` workflow
- Optionally, a separate vLLM vocal transcriber

By default MarkovSound expects ACE-Step in `~/Src/acestep.cpp`, its server at
`127.0.0.1:8085`, and model files under `~/Src/acestep.cpp/models/`. Override
these assumptions with `ACESTEP_ROOT`, `ACE_HOST`, `ACE_PORT`,
`ACE_LM_MODEL`, and `ACE_SYNTH_MODEL`.

## Quick start

```bash
git clone https://github.com/honzasvasek/MarkovSound.git
cd MarkovSound
python3 -m pip install -e ".[dev]"

# Create an isolated musical world and give its caption chain an initial voice.
./session new first-listening --use
cp runtime.example.json state/runtime.json
cp prompts/genres/techno.txt state/prompt_corpus.txt

# Generate in one terminal and listen/curate in another.
./create
./play
```

`./create` builds the initial text chain from `state/prompt_corpus.txt` and can
start the configured ACE server automatically. Replace the example corpus with
your own lines, or combine material from the supplied `techno`, `house`, and
`edm` packs before the first run.

To seed the system with actual recordings, place `.mp3`, `.wav`, `.flac`, or
`.ogg` files in `Audio/absorb/` and run:

```bash
./absorb Audio/absorb
```

Runtime audio, sessions, corpora, and serialized chains are intentionally
excluded from Git.

## The listening loop

`./play` follows the newest complete render in `Audio/queue/`. During playback:

- `INS` keeps the track and strengthens its captions in memory.
- `DEL` deletes the track and untrains its captions, lyrics, and audio codes.

Finished tracks move into the active session's `Audio/archive/`. Generation is
back-pressured by the queue unless `./create --no-wait-for-player` is used for
an intentionally unbounded training run.

## Live configuration

`state/runtime.json` is reloaded between cycles. Start from
[`runtime.example.json`](runtime.example.json):

```json
{
  "min_duration_minutes": 2,
  "max_duration_minutes": 10,
  "caption_min_words": 18,
  "caption_max_words": 70,
  "vocal_prob": 0.5,
  "takes_per_song": 3,
  "cover_strength": 0.6,
  "target_buffer_tracks": 1,
  "latent_splice_seconds": 5.0,
  "form_arc_prob": 0.75
}
```

`form_arc_prob` controls how often a new song receives a temporary dramatic
trajectory. Set it to `0.0` for unshaped Markov captions or `1.0` to shape every
new song.

## Operator commands

| Command | Purpose |
|---|---|
| `./create` | Run the autonomous generation and feedback loop. |
| `./play` | Listen to the newest queued track and curate with `INS`/`DEL`. |
| `./absorb Audio/absorb` | Learn captions and audio codes from references. |
| `./steer "prepared piano drones"` | Apply deliberate pressure to caption memory. |
| `./steer --preset techno` | Apply a supplied genre vocabulary pack. |
| `./keep TRACK.mp3` | Reinforce a generated track manually. |
| `./verwijder TRACK.mp3` | Delete and untrain a generated track manually. |
| `./session current` | Inspect the active isolated runtime session. |
| `./session save NAME --use` | Copy the current musical world and activate it. |
| `./transcriber` | Start the optional vocal transcriber. |
| `./sleep-transcriber` | Release transcriber GPU memory between jobs. |

Run `./create --help` and `./session --help` for the full command-line options.

## Development

Pure logic is covered by fast tests that do not require ACE-Step:

```bash
PYTHONPATH=src pytest -q
```

Backend integration uses the smoke scripts:

```bash
python3 scripts/smoke_ace.py
python3 scripts/smoke_vocals.py
```

The main package lives in `src/markovsound/`. Shell commands in the repository
root are intentionally thin wrappers around installable Python entry points.

## Further reading

- [`USER_GUIDE.md`](USER_GUIDE.md) — operation, curation, sessions, and runtime settings
- [`HOW_MARKOVSOUND_IMPROVES.md`](HOW_MARKOVSOUND_IMPROVES.md) — the learning and feedback model
- [`ARTISTIC_DIRECTION.md`](ARTISTIC_DIRECTION.md) — the artistic roadmap
- [`STEERING.md`](STEERING.md) — caption steering and genre packs
- [`CLAUDE.md`](CLAUDE.md) — implementation-oriented architecture notes
