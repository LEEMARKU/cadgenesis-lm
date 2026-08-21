"""
cadgenesis.transformer.self_designing.rollback
========================================
Automatic Rollback for the Self-Designing Transformer.

``AutomaticRollback`` keeps versioned snapshots of the backbone weights and
monitors a scalar metric (e.g. validation loss).  If the metric *deteriorates
beyond a tolerance* for a sustained number of consecutive checks, the system
automatically restores the best-known snapshot — guarding against bad
adaptation decisions (new architecture, grown experts, pruned layers, …).

Algorithm
---------
    snapshot(metric)              → store CPU copy of state_dict + metric
    check_and_rollback(metric)    → if metric > best·(1+tol) for `patience`
                                    consecutive checks → restore best snapshot
    rollback(snapshot_id)         → load state_dict from the snapshot

Complexity
----------
    snapshot:  O(P)   (copy of all parameters)
    rollback:  O(P)   (load_state_dict)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch.nn as nn


@dataclass
class Snapshot:
    """A versioned copy of the model weights plus its evaluation metric."""

    snapshot_id: str
    metric: float
    timestamp: float = field(default_factory=time.time)
    state_dict: dict | None = None
    arch_signature: str = ""
    metadata: dict = field(default_factory=dict)

    def size_bytes(self) -> int:
        if not self.state_dict:
            return 0
        return sum(v.numel() * v.element_size() for v in self.state_dict.values())


class AutomaticRollback:
    """
    Versioned-snapshot safety controller.

    Parameters
    ----------
    model : nn.Module
        Backbone whose weights are snapshotted / restored.
    tolerance : float
        Relative metric deterioration allowed before a check is flagged
        (e.g. 0.05 = 5% worse than the best metric).
    patience : int
        Consecutive flagged checks required before an automatic rollback fires.
    """

    def __init__(self, model: nn.Module, tolerance: float = 0.05, patience: int = 2):
        self.model = model
        self.tolerance = tolerance
        self.patience = patience

        self.snapshots: dict[str, Snapshot] = {}
        self.best_snapshot_id: str | None = None
        self.best_metric: float = float("inf")
        self._consecutive_flags = 0
        self.rollback_log: list[dict] = []
        self.history: list[float] = []

    # --------------------------------------------------------------- snapshot

    def snapshot(
        self,
        metric: float,
        arch_signature: str = "",
        metadata: dict | None = None,
    ) -> str:
        """
        Store the current weights keyed by the given metric.  Returns the id.
        """
        snapshot_id = f"snap_{len(self.snapshots)}_{int(time.time() * 1000)}"
        state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        snap = Snapshot(
            snapshot_id=snapshot_id,
            metric=float(metric),
            state_dict=state,
            arch_signature=arch_signature,
            metadata=metadata or {},
        )
        self.snapshots[snapshot_id] = snap
        self.history.append(float(metric))
        if metric < self.best_metric:
            self.best_metric = float(metric)
            self.best_snapshot_id = snapshot_id
        return snapshot_id

    # ------------------------------------------------------------ monitoring

    def check_and_rollback(
        self,
        current_metric: float,
        reason: str = "metric deterioration",
    ) -> str | None:
        """
        Feed the latest metric; return the snapshot id that was rolled back to
        (or None if no rollback was triggered).

        Rollback fires when ``current_metric`` exceeds
        ``best_metric * (1 + tolerance)`` for ``patience`` consecutive checks.
        """
        self.history.append(float(current_metric))
        threshold = self.best_metric * (1.0 + self.tolerance)

        if self.best_snapshot_id is not None and current_metric > threshold:
            self._consecutive_flags += 1
            if self._consecutive_flags >= self.patience:
                return self.rollback(self.best_snapshot_id, reason)
            return None

        # Improvement or neutral → reset the flag streak.
        self._consecutive_flags = 0
        return None

    def rollback(self, snapshot_id: str, reason: str = "manual") -> str:
        """Restore the weights of a snapshot and log the event."""
        if snapshot_id not in self.snapshots:
            raise KeyError(f"Unknown snapshot {snapshot_id!r}.")
        snap = self.snapshots[snapshot_id]
        assert snap.state_dict is not None
        self.model.load_state_dict(snap.state_dict)
        self.rollback_log.append(
            {
                "snapshot_id": snapshot_id,
                "reason": reason,
                "metric": snap.metric,
                "timestamp": snap.timestamp,
            }
        )
        return snapshot_id

    def restore(self, snapshot_id: str) -> str:
        """Explicit restore of a snapshot (no monitoring involved)."""
        return self.rollback(snapshot_id, reason="explicit_restore")

    # ---------------------------------------------------------------- status

    @property
    def best_snapshot(self) -> Snapshot | None:
        if self.best_snapshot_id is None:
            return None
        return self.snapshots.get(self.best_snapshot_id)

    def report(self) -> dict:
        return {
            "best_metric": self.best_metric,
            "best_snapshot_id": self.best_snapshot_id,
            "num_snapshots": len(self.snapshots),
            "consecutive_flags": self._consecutive_flags,
            "rollback_count": len(self.rollback_log),
        }
