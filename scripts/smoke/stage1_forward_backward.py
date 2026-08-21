"""G15 stage 1: 1-batch forward/backward on the mini preset (CPU)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from cadgenesis.smoke.stages import stage1_forward_backward

if __name__ == "__main__":
    result = stage1_forward_backward()
    print(f"status={result['status']} loss={result['loss']:.6f} "
          f"gradients_updated={result['gradients_updated']} "
          f"batch_shape={result['batch_shape']} params={result['parameters']:,} "
          f"duration_s={result['duration_s']}")