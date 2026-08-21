"""
cadgenesis.training.callbacks
=============================
Trainer callbacks for CADGenesis-LM (pure Python).

Plugins observe training/validation/checkpoint events: checkpointing,
early stopping, evaluation scheduling and metric logging.  The
``CallbackRegistry`` lets any number of plugins subscribe to one trainer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("cadgenesis.training.callbacks")


class StopTraining(Exception):
    """Raised by a callback to halt training early (early stopping)."""


@dataclass
class TrainingEvent:
    """Snapshot of training state passed to callbacks."""

    epoch: int = 0
    step: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    validation_metrics: dict[str, float] = field(default_factory=dict)
    best_validation_loss: float | None = None
    checkpoint_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "metrics": dict(self.metrics),
            "validation_metrics": dict(self.validation_metrics),
            "best_validation_loss": self.best_validation_loss,
            "checkpoint_path": self.checkpoint_path,
        }


class TrainerCallback:
    """Base class for trainer callbacks (all hooks are optional)."""

    def on_train_begin(self, event: TrainingEvent) -> None:
        """Called once before the first epoch."""

    def on_epoch_begin(self, event: TrainingEvent) -> None:
        """Called at the start of every epoch."""

    def on_step(self, event: TrainingEvent) -> None:
        """Called after each training step."""

    def on_epoch_end(self, event: TrainingEvent) -> None:
        """Called at the end of every epoch (metrics populated)."""

    def on_validation(self, event: TrainingEvent) -> None:
        """Called after validation (validation_metrics populated)."""

    def on_checkpoint(self, event: TrainingEvent) -> None:
        """Called after a checkpoint is written."""

    def on_train_end(self, event: TrainingEvent) -> None:
        """Called once after the final epoch."""


class CheckpointCallback(TrainerCallback):
    """Automatic checkpointing: every N epochs, keep-best and keep-last."""

    def __init__(
        self,
        save_checkpoint: Callable[..., Any],
        directory: str,
        every_epochs: int = 1,
        save_best: bool = True,
        save_last: bool = True,
        minimize: bool = True,
    ) -> None:
        self.save_checkpoint = save_checkpoint
        self.directory = directory
        self.every_epochs = max(1, every_epochs)
        self.save_best = save_best
        self.save_last = save_last
        self.minimize = minimize
        self.best_value: float | None = None
        self.best_path: str | None = None

    def on_epoch_end(self, event: TrainingEvent) -> None:
        if (event.epoch + 1) % self.every_epochs != 0:
            return
        if self.save_last:
            path = f"{self.directory}/last.pt"
            self.save_checkpoint(path, event.epoch, event.step, event.metrics.get("loss"))
            logger.info("checkpoint saved: %s", path)
        if self.save_best and event.validation_metrics:
            value = float(event.validation_metrics.get("loss", float("inf")))
            better = self.best_value is None or (
                value < self.best_value if self.minimize else value > self.best_value
            )
            if better:
                self.best_value = value
                self.best_path = f"{self.directory}/best.pt"
                self.save_checkpoint(self.best_path, event.epoch, event.step, value)
                logger.info("best checkpoint saved: %s (val loss %.6f)", self.best_path, value)


class EarlyStoppingCallback(TrainerCallback):
    """Stop training when validation loss plateaus."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0, minimize: bool = True) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.minimize = minimize
        self.best_value: float | None = None
        self.wait = 0

    def on_validation(self, event: TrainingEvent) -> None:
        value = float(event.validation_metrics.get("loss", float("inf")))
        better = self.best_value is None or (
            value < (self.best_value - self.min_delta)
            if self.minimize
            else value > (self.best_value + self.min_delta)
        )
        if better:
            self.best_value = value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logger.info("early stopping after %d epochs without improvement", event.epoch)
                raise StopTraining()


class MetricsLoggingCallback(TrainerCallback):
    """Log training/validation metrics to the module logger."""

    def __init__(self, logger_name: str = "cadgenesis.training") -> None:
        self.log = logging.getLogger(logger_name)

    def on_epoch_end(self, event: TrainingEvent) -> None:
        self.log.info(
            "epoch=%d step=%d loss=%.6f",
            event.epoch,
            event.step,
            event.metrics.get("loss", float("nan")),
        )

    def on_validation(self, event: TrainingEvent) -> None:
        self.log.info(
            "validation epoch=%d val_loss=%.6f best=%.6f",
            event.epoch,
            event.validation_metrics.get("loss", float("nan")),
            event.best_validation_loss or float("nan"),
        )


class MetricsJsonlCallback(TrainerCallback):
    """Persist every training/validation event to ``metrics.jsonl``.

    One JSON object per line — loss curves can be reconstructed later by
    ``scripts/plot_loss.py`` or any JSONL reader.  Fields are flattened:
    ``{"event": ..., "epoch": ..., "step": ..., <scalar metrics...>}``.
    """

    def __init__(self, directory: str, filename: str = "metrics.jsonl") -> None:
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        self.filename = filename

    def _write(self, event: TrainingEvent, kind: str, metrics: dict[str, float]) -> None:
        row: dict[str, Any] = {
            "event": kind,
            "epoch": event.epoch,
            "step": event.step,
            "best_validation_loss": event.best_validation_loss,
            "checkpoint_path": event.checkpoint_path,
            "metrics": metrics,
        }
        with (self.path / self.filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def on_train_begin(self, event: TrainingEvent) -> None:
        self._write(event, "train_begin", {})

    def on_epoch_end(self, event: TrainingEvent) -> None:
        self._write(event, "epoch_end", dict(event.metrics))

    def on_validation(self, event: TrainingEvent) -> None:
        self._write(event, "validation", dict(event.validation_metrics))

    def on_checkpoint(self, event: TrainingEvent) -> None:
        self._write(event, "checkpoint", dict(event.metrics))

    def on_train_end(self, event: TrainingEvent) -> None:
        self._write(event, "train_end", {})


class CallbackRegistry:
    """Registry of callbacks notified in registration order."""

    def __init__(self) -> None:
        self._callbacks: list[TrainerCallback] = []

    def add(self, callback: TrainerCallback) -> None:
        """Register a callback (ignores duplicates by identity)."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove(self, callback: TrainerCallback) -> None:
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def clear(self) -> None:
        self._callbacks.clear()

    def _fire(self, hook: str, event: TrainingEvent) -> None:
        for callback in self._callbacks:
            getattr(callback, hook)(event)

    def on_train_begin(self, event: TrainingEvent) -> None:
        self._fire("on_train_begin", event)

    def on_epoch_begin(self, event: TrainingEvent) -> None:
        self._fire("on_epoch_begin", event)

    def on_step(self, event: TrainingEvent) -> None:
        self._fire("on_step", event)

    def on_epoch_end(self, event: TrainingEvent) -> None:
        self._fire("on_epoch_end", event)

    def on_validation(self, event: TrainingEvent) -> None:
        self._fire("on_validation", event)

    def on_checkpoint(self, event: TrainingEvent) -> None:
        self._fire("on_checkpoint", event)

    def on_train_end(self, event: TrainingEvent) -> None:
        self._fire("on_train_end", event)

    def __len__(self) -> int:
        return len(self._callbacks)


__all__ = [
    "CallbackRegistry",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "MetricsJsonlCallback",
    "MetricsLoggingCallback",
    "StopTraining",
    "TrainerCallback",
    "TrainingEvent",
]
