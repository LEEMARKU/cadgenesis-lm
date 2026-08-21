"""
cadgenesis.training.metrics
===========================
Training metrics tracking for CADGenesis-LM (pure Python).

Tracks loss, token accuracy, perplexity, EMA loss and step counts over one
or more training/validation runs; serializable to plain dicts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricsTracker:
    """Running training metrics with exponential moving average support."""

    ema_alpha: float = 0.05

    loss_sum: float = 0.0
    correct_tokens: int = 0
    total_tokens: int = 0
    steps: int = 0
    ema_loss: float | None = None
    history: list[dict[str, float | str]] = field(default_factory=list)

    def update(self, loss: float, correct_tokens: int = 0, total_tokens: int = 0) -> None:
        """Record one training step."""
        self.loss_sum += loss
        self.steps += 1
        self.correct_tokens += int(correct_tokens)
        self.total_tokens += int(total_tokens)
        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = (1.0 - self.ema_alpha) * self.ema_loss + self.ema_alpha * loss

    def average_loss(self) -> float:
        return self.loss_sum / max(1, self.steps)

    def accuracy(self) -> float:
        if self.total_tokens <= 0:
            return 0.0
        return self.correct_tokens / self.total_tokens

    def perplexity(self) -> float:
        avg = self.average_loss()
        if avg < 0.0:
            return float("inf")
        return math.exp(avg)

    def snapshot(self) -> dict[str, float | str]:
        return {
            "steps": float(self.steps),
            "loss": round(self.average_loss(), 6),
            "ema_loss": round(self.ema_loss or 0.0, 6),
            "accuracy": round(self.accuracy(), 6),
            "perplexity": round(self.perplexity(), 6),
        }

    def record_epoch(self, tag: str = "train") -> None:
        """Push a snapshot into ``history`` for epoch-level reporting."""
        snapshot = self.snapshot()
        snapshot["tag"] = tag
        self.history.append(snapshot)

    def reset(self) -> None:
        self.loss_sum = 0.0
        self.correct_tokens = 0
        self.total_tokens = 0
        self.steps = 0
        self.ema_loss = None


def compute_accuracy(predictions: list[int], targets: list[int]) -> float:
    """Token-level accuracy over parallel prediction/target id lists."""
    if not targets:
        return 0.0
    hits = sum(1 for p, t in zip(predictions, targets, strict=False) if p == t)
    return hits / len(targets)


def log_summary(metrics: dict[str, Any]) -> str:
    """Render a metrics dict as a compact log line."""
    return ", ".join(f"{k}={v}" for k, v in metrics.items() if not isinstance(v, (list, dict)))


__all__ = ["MetricsTracker", "compute_accuracy", "log_summary"]
