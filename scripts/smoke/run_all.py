"""G15: run all four CPU smoke stages and write reports/SMOKE_TEST_RESULTS.md."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from cadgenesis.smoke.runner import main

if __name__ == "__main__":
    main()