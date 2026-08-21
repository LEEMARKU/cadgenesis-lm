"""G15 stage 3: overfit 8 records toward near-zero loss (proves learning)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from cadgenesis.smoke.stages import stage3_overfit

if __name__ == "__main__":
    result = stage3_overfit()
    print(f"status={result['status']} initial_loss={result['initial_loss']:.6f} "
          f"final_loss={result['final_loss']:.6f} target={result['target_loss']} "
          f"reached={result['target_reached']} steps={result['steps_used']} "
          f"duration_s={result['duration_s']}")