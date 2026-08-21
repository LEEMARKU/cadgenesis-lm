"""G15 stage 2: tiny dataset, 1 epoch on CPU (50 records)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from cadgenesis.smoke.stages import stage2_tiny_dataset

if __name__ == "__main__":
    result = stage2_tiny_dataset()
    print(f"status={result['status']} initial_val_loss={result['initial_val_loss']:.6f} "
          f"final_train_loss={result['final_train_loss']:.6f} "
          f"final_val_loss={result['final_val_loss']:.6f} "
          f"duration_s={result['duration_s']}")