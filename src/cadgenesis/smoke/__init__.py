"""
cadgenesis.smoke
================
Phase 12 (G15) CPU smoke stages: 1-batch forward/backward, tiny dataset
epoch, overfit proof, and a dev run with persisted loss curves.
"""

from cadgenesis.smoke.stages import (
    stage1_forward_backward,
    stage2_tiny_dataset,
    stage3_overfit,
    stage4_dev_run,
)
from cadgenesis.smoke.runner import run_all

__all__ = [
    "run_all",
    "stage1_forward_backward",
    "stage2_tiny_dataset",
    "stage3_overfit",
    "stage4_dev_run",
]