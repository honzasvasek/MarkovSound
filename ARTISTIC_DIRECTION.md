# Artistic Direction

MarkovSound already has an unusual strength: it remembers what it made in
three different languages (description, lyrics, and audio codes). Its main
artistic weakness is larger-scale intention. A caption can be vivid while the
resulting track still behaves like a single static prompt, and related takes
can feel like retries instead of transformations.

The next development should preserve the accidents and glitches while giving
them consequences over time.

## 1. Compositional dramaturgy — implemented

- Keep the sampled Markov caption as source material.
- Choose a genre-independent form arc for some new songs.
- Add explicit beginning/middle/end behavior only to the synthesis prompt.
- Reuse the arc across related cover takes, changing the take-level treatment.
- Store source caption, expanded prompt, and form metadata separately so human
  feedback does not merely teach the chain its own hand-written instructions.

This is controlled live with `form_arc_prob` in `state/runtime.json`.

## 2. Coherent variation between takes

Treat a run of cover takes as a miniature lineage. Preserve selected anchors
(tempo, metre, or tonal center), then mutate one dimension at a time. Record
the mutation explicitly in the sidecar so a listener can tell whether a change
was intentional and so later curation can act on it.

## 3. Memory of recent contrast

Read a small window of recent sidecars before planning a new song. Bias away
from recently repeated tempo bands, metres, form arcs, and key centers without
hard-blocking them. This should be a decaying novelty pressure, not a shuffle
rule: recurrence remains possible, but immediate sameness becomes less likely.

## 4. Let taste select strategies

Extend keep/delete feedback from captions to planning metadata. Kept tracks
should gently raise the weight of their form and mutation strategy; deleted
tracks should lower it. Keep those weights bounded and periodically flattenable
so preference creates a recognizable practice without collapsing exploration.

## 5. Align audio-code discontinuities with form

The code sampler already detects dead-end jumps and can splice latent segments.
Use the selected form arc to place or interpret those discontinuities: a rupture
may favor one strong boundary, erosion may shorten or thin later segments, and
unstable orbit may permit several returning boundaries. This phase needs live
ACE smoke tests because it changes the rendered audio rather than prompt logic
alone.

## Evaluation

For each phase, compare short blind batches on three questions: can a listener
describe the track's change over time, do related takes feel connected but not
duplicated, and does the output remain surprising? Sidecar metadata should make
every planning choice inspectable without pretending that those choices alone
measure artistic quality.
