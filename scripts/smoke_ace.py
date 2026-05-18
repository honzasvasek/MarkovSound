"""Smoke test: start ace-server, generate one short track from a hardcoded caption."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from markovsound.ace_client import ensure_server, lm, shutdown_server, synth
from markovsound.config import AceConfig


def main() -> int:
    cfg = AceConfig.discover()
    proc = ensure_server(cfg)
    try:
        caption = (
            "Bright two-tone ska with upbeat horn section, choppy guitar, "
            "walking bass, energetic male vocals, irresistible dancing tempo."
        )
        request = {
            "caption": caption,
            "duration": 60,
            "vocal_language": "en",
            "inference_steps": 8,
            "guidance_scale": 1.0,
            "shift": 3.0,
        }
        t0 = time.time()
        enriched = lm(cfg, request)
        t1 = time.time()
        print(f"  /lm took {t1 - t0:.1f}s")
        print(f"  enriched keys: {sorted(enriched.keys())}")
        print(f"  bpm={enriched.get('bpm')} key={enriched.get('keyscale')} dur={enriched.get('duration')}")
        if enriched.get("lyrics"):
            print(f"  lyrics (first 200): {enriched['lyrics'][:200]!r}")

        t2 = time.time()
        mp3, latent = synth(cfg, enriched)
        t3 = time.time()
        print(f"  /synth took {t3 - t2:.1f}s")
        print(f"  mp3 bytes: {len(mp3)}  latent bytes: {len(latent) if latent else None}")

        out_dir = Path(__file__).resolve().parents[1] / "Audio"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_mp3 = out_dir / "smoke.mp3"
        out_mp3.write_bytes(mp3)
        print(f"  wrote {out_mp3}")
        return 0
    finally:
        shutdown_server(proc)


if __name__ == "__main__":
    sys.exit(main())
