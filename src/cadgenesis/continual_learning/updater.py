"""cadgenesis.continual_learning.updater
=====================================
Incremental model updater for continual-learning checkpoints.

Each save writes the model ``state_dict`` to ``task_<id>_<version>.pt`` (via a
temporary file + ``os.replace`` for atomicity) and a JSON sidecar holding the
task metadata (``task_id``, ``step``, ``version``, ``timestamp``).  Versions
bump per save, derived from the checkpoints already on disk.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_VERSION_RE = re.compile(r"^task_.+_(\d+)\.pt$")


class ModelUpdater:
    """Persists and reloads incremental task checkpoints."""

    def save_incremental(
        self,
        model: nn.Module,
        task_id: str,
        base_dir: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save ``state_dict`` plus a JSON sidecar; return the checkpoint path.

        ``metadata`` may carry arbitrary keys (``step`` is picked up by
        default); ``task_id``, ``version`` and ``timestamp`` always win.
        """
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        version = self._next_version(base, task_id)
        path = self.checkpoint_path(base, task_id, version)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, path)
        meta: dict[str, Any] = dict(metadata or {})
        step = int(meta.get("step", 0))
        meta["task_id"] = task_id
        meta["step"] = step
        meta["version"] = version
        meta["timestamp"] = datetime.now().isoformat(timespec="seconds")
        sidecar = path.with_suffix(".json")
        with sidecar.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        return path

    def load_latest(self, base_dir: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Load the highest-version checkpoint in ``base_dir`` (any task)."""
        checkpoints = self.list_checkpoints(base_dir)
        if not checkpoints:
            raise FileNotFoundError(f"no checkpoints found in {base_dir}")
        latest = checkpoints[-1]
        return self._load(str(latest["path"])), latest

    def load_latest_for(
        self, base_dir: str | Path, task_id: str
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Load the highest-version checkpoint for one task."""
        matches = [c for c in self.list_checkpoints(base_dir) if c["task_id"] == task_id]
        if not matches:
            raise FileNotFoundError(f"no checkpoints for task {task_id!r} in {base_dir}")
        latest = matches[-1]
        return self._load(str(latest["path"])), latest

    def list_checkpoints(self, base_dir: str | Path) -> list[dict[str, Any]]:
        """All checkpoints, ascending by ``(version, path)``.

        Each entry is the JSON sidecar metadata (synthesized from the filename
        when the sidecar is missing) plus a ``path`` key.
        """
        base = Path(base_dir)
        if not base.is_dir():
            return []
        checkpoints: list[dict[str, Any]] = []
        for path in base.glob("task_*.pt"):
            sidecar = path.with_suffix(".json")
            if sidecar.is_file():
                with sidecar.open("r", encoding="utf-8") as fh:
                    meta: dict[str, Any] = json.load(fh)
            else:
                match = _VERSION_RE.match(path.name)
                meta = {
                    "task_id": path.name[5:].rsplit("_", 1)[0],
                    "version": int(match.group(1)) if match else 0,
                }
            meta["path"] = str(path)
            checkpoints.append(meta)
        checkpoints.sort(key=lambda m: (int(m.get("version", 0)), str(m["path"])))
        return checkpoints

    def checkpoint_path(self, base_dir: str | Path, task_id: str, version: int) -> Path:
        """Concrete checkpoint path for a task and version."""
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)
        return Path(base_dir) / f"task_{safe_id}_{version}.pt"

    def _next_version(self, base: Path, task_id: str) -> int:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)
        candidates = (p.stem.rsplit("_", 1)[-1] for p in base.glob(f"task_{safe_id}_*.pt"))
        versions = [int(tail) for tail in candidates if tail.isdigit()]
        return max(versions, default=0) + 1

    @staticmethod
    def _load(path: str) -> dict[str, torch.Tensor]:
        return torch.load(path, map_location="cpu", weights_only=True)


__all__ = ["ModelUpdater"]
