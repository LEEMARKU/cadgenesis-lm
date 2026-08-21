"""G15 stage 4: dev run (200 records, few epochs) with persisted loss curve."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from cadgenesis.smoke.stages import SMOKE_OUTPUTS, stage4_dev_run

if __name__ == "__main__":
    result = stage4_dev_run(out_dir=SMOKE_OUTPUTS / "stage4")
    print(f"status={result['status']} final_train_loss={result['final_train_loss']:.6f} "
          f"final_val_loss={result['final_val_loss']:.6f} "
          f"best_val_loss={result['best_val_loss']:.6f} "
          f"metrics={result['metrics_path']} ckpt={result['checkpoint_path']} "
          f"duration_s={result['duration_s']}")