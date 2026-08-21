"""
cadgenesis.training.checkpoint
==============================
Automatic checkpointing and resume support for CADGenesis-LM.

``CheckpointManager`` writes periodic checkpoints plus a lightweight
``meta.json`` (epoch/step/best loss/version) so training can resume
exactly where it left off, with retention of the N best checkpoints.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("cadgenesis.training.checkpoint")

META_FILE = "meta.json"


class CheckpointManager:
    """Manage automatic checkpoints + resume metadata for a training run."""

    def __init__(
        self,
        directory: str,
        save_checkpoint: Callable[[str, int, int, float | None], Any] | None = None,
        every_steps: int = 0,
        every_epochs: int = 0,
        keep_best: int = 3,
        minimize: bool = True,
    ) -> None:
        self.directory = directory
        self.save_checkpoint = save_checkpoint
        self.every_steps = max(0, every_steps)
        self.every_epochs = max(0, every_epochs)
        self.keep_best = max(1, keep_best)
        self.minimize = minimize
        self.best: list[tuple[float, str]] = []
        self.last_path: str | None = None
        self.last_epoch = -1
        self.last_step = -1
        self.best_loss: float | None = None
        os.makedirs(directory, exist_ok=True)

    # ------------------------------------------------------------- decisions

    def should_checkpoint(self, step: int, epoch: int) -> bool:
        if self.every_steps > 0 and step > 0 and step % self.every_steps == 0:
            return True
        return self.every_epochs > 0 and (epoch + 1) % self.every_epochs == 0

    # ---------------------------------------------------------------- saving

    def save(
        self,
        epoch: int,
        step: int,
        metrics: dict[str, Any] | None = None,
        validation_loss: float | None = None,
        path: str | None = None,
    ) -> str:
        """Write a checkpoint; path defaults to ``checkpoint-{epoch}-{step}.pt``."""
        target = path or os.path.join(self.directory, f"checkpoint-{epoch}-{step}.pt")
        if self.save_checkpoint is not None:
            self.save_checkpoint(target, epoch, step, validation_loss)
        else:
            self._touch(target)
        self.last_path = target
        self.last_epoch = epoch
        self.last_step = step
        if validation_loss is not None:
            self._track_best(validation_loss, target)
            better = self.best_loss is None or (
                (validation_loss < self.best_loss)
                if self.minimize
                else (validation_loss > self.best_loss)
            )
            if better:
                self.best_loss = validation_loss
        self._write_meta()
        logger.info("checkpoint written: %s (epoch=%d step=%d)", target, epoch, step)
        return target

    def save_best(
        self,
        epoch: int,
        step: int,
        validation_loss: float,
        load_checkpoint: Callable[[str], Any] | None = None,
    ) -> str:
        """Snapshot the current best weights to ``best.pt``."""
        source = self.last_path or self._latest()
        target = os.path.join(self.directory, "best.pt")
        if source and load_checkpoint is not None and os.path.exists(source):
            checkpoint = load_checkpoint(source)
            saved = getattr(checkpoint, "save", None)
            if callable(saved):
                saved(target)
            else:
                self._touch(target)
        else:
            self._touch(target)
        self._track_best(validation_loss, target)
        self.best_loss = validation_loss
        self._write_meta()
        return target

    # ---------------------------------------------------------------- resume

    def resume_from(self) -> str | None:
        """Path of the most recent checkpoint (for ``--resume-from``)."""
        return self.last_path or self._latest()

    def load_meta(self) -> dict[str, Any]:
        meta_path = os.path.join(self.directory, META_FILE)
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as handle:
                return json.load(handle)
        return {}

    def _latest(self) -> str | None:
        candidates = [
            os.path.join(self.directory, name)
            for name in os.listdir(self.directory)
            if name.endswith(".pt")
        ]
        if not candidates:
            return None
        return max(candidates, key=os.path.getmtime)

    # --------------------------------------------------------------- internal

    def _track_best(self, value: float, path: str) -> None:
        self.best.append((value, path))
        self.best.sort(key=lambda pair: pair[0] if self.minimize else -pair[0])
        for _, stale in self.best[self.keep_best :]:
            if os.path.exists(stale) and not stale.endswith("best.pt"):
                os.remove(stale)
        self.best = self.best[: self.keep_best]

    def _touch(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("placeholder")  # real state written by the trainer

    def _write_meta(self) -> None:
        meta = {
            "last_epoch": self.last_epoch,
            "last_step": self.last_step,
            "best_loss": self.best_loss,
            "best": [[value, path] for value, path in self.best],
            "last_path": self.last_path,
            "saved_at": time.time(),
        }
        meta_path = os.path.join(self.directory, META_FILE)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)


def cleanup_checkpoints(directory: str, keep: int = 3) -> list[str]:
    """Delete oldest ``checkpoint-*.pt`` files beyond ``keep`` (keeps meta.json)."""
    files = sorted(
        (
            os.path.join(directory, name)
            for name in os.listdir(directory)
            if name.startswith("checkpoint-") and name.endswith(".pt")
        ),
        key=os.path.getmtime,
    )
    removed: list[str] = []
    for stale in files[: max(0, len(files) - keep)]:
        os.remove(stale)
        removed.append(stale)
    return removed


def move_checkpoint(source: str, destination_dir: str) -> str:
    """Move a checkpoint into ``destination_dir`` (archive support)."""
    os.makedirs(destination_dir, exist_ok=True)
    target = os.path.join(destination_dir, os.path.basename(source))
    shutil.move(source, target)
    return target


__all__ = ["META_FILE", "CheckpointManager", "cleanup_checkpoints", "move_checkpoint"]
