# How MarkovSound Improves

MarkovSound does not improve by optimizing toward one fixed target. It improves
by accumulating memory, listening back to what it actually made, accepting
human preference, and periodically removing the habits that make it dull.

The system is closer to a workshop than a trained model: every cycle leaves
material behind that changes what later cycles can do.

## The short version

```text
seed material
    ↓
generate a song concept
    ↓
render audio
    ↓
listen back to the result
    ↓
train memory from what was actually heard
    ↓
human keeps / deletes / steers / cleans
    ↓
future generations shift
```

A useful way to think about it:

- **absorb** gives it vocabulary from outside examples
- **autofeedback** gives it memory of its own output
- **keep** says “more like this”
- **verwijder** says “less like this”
- **steer** gives deliberate directional pressure
- **dedup / cleanup tools** restore diversity when memory becomes too sticky

## Three different kinds of memory

### 1. Caption memory

The text chain stores descriptive language about music: textures, instruments,
movement, density, mood, and arrangement. It proposes the next song concept.

After a track is rendered, `ace-understand` describes what was actually heard.
That description — cleaned to prose only — can be trained back into the text
chain. This matters because the system learns from **realized output**, not only
from what it originally intended to ask for.

### 2. Lyric memory

The lyric chain learns structured lyric material: section tags, stage
directions, syllables, repeated phrases, and fragments of language. It can grow
from:

- the lyricist vocabulary feeder
- lyrics that ACE produced
- lyrics the transcriber heard back from generated songs
- manual maintenance such as `./replace-lyrics`

Because it remembers structure as well as words, it can gradually develop a
house style of vocal behavior, not merely a word list.

### 3. Audio-code memory

The codes chain stores ACE audio-token trajectories learned from absorbed or
generated audio. When mature enough, it can bypass the language-model planning
step and sample audio-code sequences directly. This is one of the system’s
routes toward stranger, less conventionally “songwriterly” output.

## Improvement happens through feedback, not just accumulation

Blind accumulation would eventually make the system repetitive or polluted.
MarkovSound therefore has several corrective loops.

### Self-feedback

For each generated track, the system can:

- understand the audio
- transcribe vocals when available
- train codes from what was heard
- train lyrics from the best available heard lyric source
- train caption memory from cleaned descriptive prose

That is a form of reality check: the chain drifts toward what the system is
actually capable of producing, not just toward its prompts.

### Human feedback

While listening in `./play`:

- `INS` keeps a track and boosts its caption weight
- `DEL` removes a track and untrains its caption, codes, and lyrics

These are not ratings after the fact; they directly alter future probability.
Repeated human preference becomes style pressure.

### Deliberate steering

`./steer` injects chosen vocabulary with high weight. This is useful when the
system has drifted into a region that is coherent but not interesting, or when
you want to pull it toward a new genre or texture without rebuilding from zero.

### Hygiene and repair

Some learned material is useful once but harmful as a permanent habit. The
maintenance tools exist because improvement also means forgetting or rewriting:

- `dedup` flattens overweight transitions when output becomes repetitive
- `replace-lyrics` rewrites lyric-chain vocabulary live
- `clean-caption-metadata` removes old metadata suffix paths that pollute prose
- timestamp pruning removes lyric artifacts that should never become style

## Why this is different from a normal playlist generator

A normal generator emits independent tracks from a static model. MarkovSound is
stateful:

- what it made yesterday affects what it can make today
- what you kept affects what it is likely to revisit
- what you deleted becomes less likely
- what it misheard can become future material unless corrected
- what is overlearned can be flattened or replaced

That makes improvement aesthetic rather than purely numerical. The system gets
better when its memory becomes more specific, more alive, and less generic —
while still retaining enough entropy to surprise itself.

## What “better” means here

For this project, improvement does **not** mean “more polished pop songs.” It
means some combination of:

- more distinctive vocabulary
- fewer generic ACE defaults
- richer recurrence across songs
- stronger alignment with human taste
- enough diversity to avoid collapse
- productive artifacts rather than accidental boilerplate

The process is intentionally reversible. You can seed, steer, keep, delete,
replace, flatten, and rebuild. MarkovSound improves because its memory remains
editable.
