# Steering MarkovSound toward a genre

The text chain (`state/markov_chain.pkl`) drifts in whatever direction
autofeedback pushes it. To pull it toward a target style, train it
heavy-weight with seed captions for that style.

## One-shot: ad-hoc steer

    ./steer "deep house with rolling bassline and soulful vocal sample"

That trains the caption x3 at weight 5.0, and auto-whitelists any
pop-blocklist words it contains (so autofeedback won't reject future
captions containing them).

## Presets (genre packs)

Caption packs live in `prompts/genres/<NAME>.txt`, one caption per
line. Apply one or more:

    ./steer --preset edm
    ./steer --preset techno --preset house

Each line is trained individually (so the chain learns the transitions
*inside* each caption as a separate unit, not as one giant blob), then
the whole vocabulary gets whitelisted in one batch.

Existing presets:

- `edm` — big-room / festival / progressive / melodic EDM
- `techno` — deep, industrial, minimal, acid, dub, melodic techno
- `house` — classic, deep, chicago, funky, tech, soulful, afro, garage, progressive, lo-fi

Add a new genre by dropping a new `prompts/genres/<NAME>.txt` with ~10
captions and `./steer --preset <NAME>`.

## What else to tune when changing genre

- `src/markovsound/loop.py:_LM_NEGATIVE_PROMPT` — global anti-cheese
  list sent to `/lm`. Don't put your target genre in here. Currently
  rejects polish/anthemic/singalong only, leaving room for EDM/dance/jazz/etc.
- `src/markovsound/describe.py:_POP_WORDS` / `_POP_PHRASES` — caption
  blocklist that autofeedback respects. `./steer` auto-whitelists
  whatever is in your seed captions; bulk-whitelist via
  `state/steering.json` if you want to bypass it permanently.
- `--vocal-prob N` on `./create` — most techno is instrumental; you may
  want to lower this from 0.8 to ~0.3 when steering toward techno.
- Restart `./create` after editing any of these so the loop picks them
  up (the running process holds the imports in memory).

## Verify the steer is working

After `./steer` and a `./create` restart, watch the new captions:

    grep '^\[..:..:..\] caption:' /tmp/markovsound-create.log | tail -10

You should see the seed vocabulary surface in the generated captions
within a few cycles.
