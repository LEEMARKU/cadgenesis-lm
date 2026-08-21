"""
cadgenesis.training.scheduler
=============================
Learning-rate scheduler factory for CADGenesis-LM.

Single factory ``build_scheduler`` providing linear warmup, cosine
annealing (with optional restarts), linear decay, constant schedules,
and the modern **WSD** (warmup-stable-decay, MiniCPM / DeepSeek-style)
schedule; plus a no-op scheduler for pure-Python runs.
"""

from __future__ import annotations

import math
from typing import Any

import torch

try:
    import torch
except ImportError:  # pragma: no cover - torch optional
    torch = None  # type: ignore[assignment]

SCHEDULES = (
    "constant",
    "linear_warmup",
    "cosine",
    "cosine_restarts",
    "linear_decay",
    "wsd",
)


def _linear_warmup(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(step + 1) / float(warmup_steps))


def _cosine_decay(step: int, total_steps: int) -> float:
    progress = min(1.0, max(0.0, float(step) / max(1, total_steps)))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _linear_decay(step: int, total_steps: int) -> float:
    return max(0.0, 1.0 - float(step) / max(1, total_steps))


def wsd_lr_scale(
    step: int,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """
    WSD (warmup-stable-decay) LR multiplier.

    * warmup:    linear 0 → 1 over ``warmup_steps``
    * stable:    constant 1.0 for ``stable_steps``
    * decay:     cosine 1.0 → ``min_lr_ratio`` over ``decay_steps``

    After the decay window the LR is held at ``min_lr_ratio`` (annealing
    may be re-triggered by re-initialising with a fresh decay window, which
    is the "second pretraining phase" trick from MiniCPM).
    """
    if step < warmup_steps:
        return _linear_warmup(step, warmup_steps)
    step -= warmup_steps
    if step < stable_steps:
        return 1.0
    step -= stable_steps
    if step < decay_steps:
        progress = float(step) / max(1, decay_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio


def build_wsd_scheduler(
    optimizer: Any,
    num_train_steps: int,
    warmup_steps: int = 0,
    stable_ratio: float = 0.75,
    decay_ratio: float = 0.25,
    min_lr_ratio: float = 0.1,
) -> Any:
    """
    Build a WSD (warmup-stable-decay) scheduler.

    ``stable_ratio + decay_ratio`` should equal 1.0 (fractions of the steps
    *after* warmup).  MiniCPM-style: train at peak LR, anneal at the end,
    and optionally repeat (re-decay) for better checkpoints.
    """
    if torch is None:
        return _NoOpScheduler()
    if not 0.0 <= stable_ratio <= 1.0 or not 0.0 <= decay_ratio <= 1.0:
        raise ValueError("stable_ratio / decay_ratio must be in [0, 1].")
    if not 0.0 < min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in (0, 1].")
    total = max(1, num_train_steps - max(0, warmup_steps))
    stable_steps = round(total * stable_ratio)
    decay_steps = max(1, total - stable_steps)

    def lr_lambda(step: int) -> float:
        return wsd_lr_scale(
            step,
            warmup_steps=max(0, warmup_steps),
            stable_steps=stable_steps,
            decay_steps=decay_steps,
            min_lr_ratio=min_lr_ratio,
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_scheduler(
    optimizer: Any,
    schedule: str = "cosine",
    num_train_steps: int = 1000,
    warmup_steps: int = 0,
    total_steps: int | None = None,
    restart_interval: int | None = None,
    wsd_stable_ratio: float = 0.75,
    wsd_decay_ratio: float = 0.25,
    wsd_min_lr_ratio: float = 0.1,
) -> Any:
    """Build an LR scheduler around ``optimizer``.

    Parameters
    ----------
    schedule : one of :data:`SCHEDULES`
    num_train_steps : total steps for decay schedules
    warmup_steps : linear warmup applied before the main schedule
    total_steps : optional override for ``num_train_steps``
    restart_interval : for ``cosine_restarts``, steps per restart cycle
    wsd_stable_ratio / wsd_decay_ratio / wsd_min_lr_ratio : WSD parameters
    """
    if torch is None:
        return _NoOpScheduler()
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule {schedule!r}; expected one of {SCHEDULES}")
    if schedule == "wsd":
        return build_wsd_scheduler(
            optimizer,
            num_train_steps=num_train_steps,
            warmup_steps=warmup_steps,
            stable_ratio=wsd_stable_ratio,
            decay_ratio=wsd_decay_ratio,
            min_lr_ratio=wsd_min_lr_ratio,
        )
    total = total_steps if total_steps is not None else max(1, num_train_steps)
    if restart_interval is not None and restart_interval > 0:
        total = restart_interval
    effective_warmup = max(0, warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < effective_warmup:
            return _linear_warmup(step, effective_warmup)
        decay_step = step - effective_warmup
        if schedule == "constant":
            return 1.0
        if schedule == "linear_warmup":
            return 1.0
        if schedule == "cosine":
            return _cosine_decay(decay_step, max(1, total))
        if schedule == "cosine_restarts":
            cycle = decay_step // max(1, total)
            within = decay_step - cycle * total
            return _cosine_decay(within, max(1, total))
        if schedule == "linear_decay":
            return _linear_decay(decay_step, max(1, total))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class _NoOpScheduler:
    """Scheduler stand-in when torch is unavailable (no-op steps)."""

    def __init__(self) -> None:
        self.last_epoch = 0

    def step(self, *args: Any, **kwargs: Any) -> None:
        self.last_epoch += 1

    def state_dict(self) -> dict[str, Any]:
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.last_epoch = int(state.get("last_epoch", 0))


__all__ = ["SCHEDULES", "build_scheduler"]
