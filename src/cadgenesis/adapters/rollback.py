"""cadgenesis.adapters.rollback
===========================
Adapter / model rollback.

Checkpoint-based rollback: snapshots a model's state_dict to
``<base_dir>/<adapter_id>_<timestamp>.pt`` and restores it on demand.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class AdapterRollback:
    """Snapshot and restore adapter / model weights from checkpoint files."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else None

    def snapshot(
        self,
        model: nn.Module,
        adapter_id: str,
        base_dir: str | Path | None = None,
        reason: str = "",
    ) -> str:
        """Save ``model`` weights to a timestamped checkpoint; returns its id."""
        base = Path(base_dir) if base_dir is not None else self.base_dir
        if base is None:
            raise ValueError("base_dir is required (pass it or set it at construction)")
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{adapter_id}_{time.time_ns()}.pt"
        payload: dict[str, Any] = {
            "state_dict": model.state_dict(),
            "adapter_id": adapter_id,
            "reason": reason,
            "timestamp": time.time(),
        }
        torch.save(payload, path)
        return path.name

    def rollback(
        self,
        adapter_id: str,
        checkpoint_id: str | None = None,
        base_dir: str | Path | None = None,
        model: nn.Module | None = None,
    ) -> str:
        """Restore the adapter's weights; returns the checkpoint path used."""
        base = Path(base_dir) if base_dir is not None else self.base_dir
        if base is None:
            raise ValueError("base_dir is required (pass it or set it at construction)")
        if checkpoint_id is None:
            checkpoints = self.list_checkpoints(adapter_id, base)
            if not checkpoints:
                raise FileNotFoundError(f"no checkpoints found for adapter {adapter_id!r}")
            path = Path(checkpoints[-1])
        else:
            candidate = Path(checkpoint_id)
            path = candidate if candidate.is_absolute() else base / candidate
        if not path.exists():
            raise FileNotFoundError(f"checkpoint {str(path)!r} does not exist")

        payload = _load_checkpoint(path)
        if model is not None:
            model.load_state_dict(payload["state_dict"])
        return str(path)

    def list_checkpoints(self, adapter_id: str, base_dir: str | Path | None = None) -> list[str]:
        """Checkpoint paths for ``adapter_id``, ordered by creation time."""
        base = Path(base_dir) if base_dir is not None else self.base_dir
        if base is None:
            raise ValueError("base_dir is required (pass it or set it at construction)")
        checkpoints = sorted(
            base.glob(f"{adapter_id}_*.pt"),
            key=lambda p: int(p.name.rsplit("_", 1)[1].removesuffix(".pt")),
        )
        return [str(path) for path in checkpoints]


def _load_checkpoint(path: Path) -> dict[str, Any]:
    """torch.load with a weights_only fallback for older torch versions."""
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        pass
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)
