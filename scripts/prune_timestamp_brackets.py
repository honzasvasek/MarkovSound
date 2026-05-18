"""Compatibility wrapper for the packaged prune-timestamps CLI."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from markovsound.cli_prune_timestamps import main


if __name__ == "__main__":
    sys.exit(main())
