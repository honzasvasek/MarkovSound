"""Force a vocal cycle end-to-end, then ask /understand whether it heard vocals.

Bypasses the loop's slot rotation so the user's running ./create can't
overwrite the test artifacts.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from markovsound.ace_client import lm, server_alive, synth, understand
from markovsound.config import AceConfig, Paths
from markovsound.loop import _vocalize_caption
from markovsound.lyrics_markov import load_lyrics_chain, sample_lyrics, word_ratio
from markovsound.markov import generate_caption, load_chain


def main() -> int:
    cfg = AceConfig.discover()
    paths = Paths.discover()
    if not server_alive(cfg):
        print("ace-server not running; start ./create first or run ace-server manually")
        return 1

    chain, order = load_chain(paths.chain_path)
    lyrics_chain, lyrics_order = load_lyrics_chain(paths.lyrics_chain_path)
    out_dir = paths.audio_dir / "vocal_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_tests = 3
    results = []
    for i in range(n_tests):
        print(f"\n=== smoke test {i + 1}/{n_tests} ===")
        caption = generate_caption(chain, order, min_words=18, max_words=70, temperature=1.0)
        print(f"caption raw:      {caption}")
        cap = _vocalize_caption(caption, "en")
        print(f"caption vocal:    {cap}")
        lyrics = sample_lyrics(lyrics_chain, lyrics_order, min_tokens=80, max_tokens=400)
        if not lyrics:
            print("lyrics chain failed to produce a usable sample; using ''")
            lyrics = ""
        else:
            print(f"lyrics ratio:     {word_ratio(lyrics):.2f}")
            print(f"lyrics (head):    {lyrics[:200]}")

        req = {
            "caption": cap,
            "duration": 30,
            "vocal_language": "en",
            "lyrics": lyrics,
            "lm_cfg_scale": 4.0,
            "inference_steps": 8,
            "guidance_scale": 1.0,
            "shift": 3.0,
        }
        t0 = time.time()
        enriched = lm(cfg, req)
        print(f"/lm done in {time.time() - t0:.1f}s")
        mp3, _latent = synth(cfg, enriched)
        out = out_dir / f"smoke_{i + 1}.mp3"
        out.write_bytes(mp3)
        print(f"wrote {out} ({len(mp3)} bytes)")

        meta, _ = understand(cfg, out)
        heard_lyrics = (meta.get("lyrics") or "").strip()
        heard_lang = meta.get("vocal_language") or ""
        # Count word vs bracket
        import re
        words = len(re.findall(r"[A-Za-z']+", heard_lyrics))
        non_bracket_words = len(re.findall(r"(?<!\[)[A-Za-z']+", re.sub(r"\[[^\]]*\]", "", heard_lyrics)))
        print(f"/understand says: vocal_lang={heard_lang!r} heard_word_count≈{non_bracket_words}")
        print(f"  heard lyrics (head): {heard_lyrics[:200]}")
        results.append({
            "i": i + 1,
            "caption": cap,
            "heard_words": non_bracket_words,
            "heard_lang": heard_lang,
            "heard_lyrics_head": heard_lyrics[:200],
        })

    print("\n=== summary ===")
    for r in results:
        verdict = "VOCALS ✓" if r["heard_words"] >= 8 else "≈silent"
        print(f"  test {r['i']}: {verdict}  words≈{r['heard_words']}  lang={r['heard_lang']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
