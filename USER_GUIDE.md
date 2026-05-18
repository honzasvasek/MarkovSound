# 🎧 MarkovSound: User Guide

MarkovSound is an autonomous generative music system: it composes tracks, listens back to them, and changes future output through its own feedback plus your curation. For the deeper learning model, see `HOW_MARKOVSOUND_IMPROVES.md`.

## 🌀 How it Works

The system is built on a **Feedback Loop**. It uses Markov chains (statistical models of sequences) to generate instructions for a diffusion model (ACE-Step), which then synthesizes audio.

### The Pipeline:
1. **Generation**: The system samples a descriptive caption from its internal memory (Markov chain).
2. **Synthesis**: This caption is sent to the ACE-server, which generates an MP3.
3. **Analysis (Autofeedback)**: The system listens back using `ace-understand` and, when available, a separate vocal transcriber.
4. **Learning**: Clean descriptive prose, heard lyrics, and audio codes feed the relevant memories so later cycles are shaped by what was actually produced.

### The Three Chains:
The system manages three distinct "memories":
- **Text Chain**: Generates the conceptual descriptions (captions).
- **Codes Chain**: Generates raw audio tokens, bypassing the language model to create more experimental, non-human sonic structures.
- **Lyrics Chain**: Generates lyrics for vocal tracks.

---

## 🕹️ Steering the System

MarkovSound is designed to be steered, not just run. You can influence its evolution in several ways:

### 1. Human Curation (The Quick Loop)
While listening with the `./play` script, you can curate the system in real-time using your keyboard:
- **`INS` (Insert/Keep)**: Tells the system "I like this." This boosts the track's influence and encourages the system to explore similar sonic territories.
- **`DEL` (Delete/Remove)**: Tells the system "This is not it." The track is deleted and its contribution is untrained from the chain, pushing the system away from that sound.

### 2. Diversity & Entropy (`dedup`)
Over time, the Markov chains can become "stuck" on certain patterns (peaks in weight). 
- Running the **`dedup`** tool flattens these weights.
- This resets the "steering" and forces the system to be more diverse and experimental again.

### 3. The Seed (Corpus)
The system starts its life from a **corpus** of text. By adding your own descriptive phrases, genres, or avant-garde concepts to the initial training data, you set the starting trajectory of the evolution.

### 4. Live Runtime Config
`./create` reads `state/runtime.json` before every cycle, so these values can be edited while it is running — no restart required:

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
  "latent_splice_seconds": 5.0
}
```

For example, change `min_duration_minutes` and `max_duration_minutes` to `4` and `6`, save the file, and the next **new song** will use that range. Existing multi-take songs keep their already chosen duration until the next song starts. `target_buffer_tracks` controls how many unplayed tracks `./create` keeps ready in `Audio/queue/` while `./play` consumes the oldest queued item. A tracked template lives at `runtime.example.json`.

`--codes-mode` and `--lyrics-mode` are still startup flags because they change generation strategy rather than per-song shape.

---

## 🧪 Development Checks

For fast local regression checks that do not need ACE-Step running:

```bash
python3 -m pip install -e ".[dev]"
PYTHONPATH=src pytest -q
```

The pytest suite covers pure logic such as live runtime config validation, prompt shaping, Markov-chain helpers, and cycle carry-over behavior. ACE-server integration still uses the smoke scripts because it depends on a live model backend.

## 🛠️ Common Commands

| Command | Description |
| :--- | :--- |
| `./create` | Starts the main generation loop. |
| `./play` | Launches the interactive player with `INS`/`DEL` curation. |
| `PYTHONPATH=src python3 -m markovsound.cli_dedup` | Flattens chains to increase diversity and break repetitions. |
| `./replace-lyrics OLD NEW` | Rewrites lyric-chain vocabulary. |
| `./clean-caption-metadata` | Removes legacy metadata suffix paths from the text chain. |
| `./verwijder <file>` | Manually remove a track and untrain it. |
| `./keep <file>` | Manually boost a track's influence. |

## 🎼 Philosophy
MarkovSound is intended to move away from "pop" and commercial music structures. It aims for the intersection of contemporary composition, free jazz, and algorithmic exploration. The "artefacts" and "glitches" of the process are not bugs—they are the primary material.
