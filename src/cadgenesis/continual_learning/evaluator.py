"""cadgenesis.continual_learning.evaluator
=======================================
Continual-learning evaluation (per-task accuracy, forgetting).

Forgetting for a task is the standard CL metric: the difference between the
best accuracy measured so far and the latest one.  A single measurement (or an
unknown task) yields ``0.0`` — there is no prior best to forget against.
"""

from __future__ import annotations

from typing import Any


class ContinualEvaluator:
    """Tracks per-task accuracy histories and derives forgetting metrics."""

    def __init__(self) -> None:
        self._history: dict[str, list[float]] = {}

    def record_task_acc(self, task_id: str, acc: float) -> None:
        """Append one accuracy measurement for ``task_id``."""
        self._history.setdefault(task_id, []).append(float(acc))

    def accuracy_history(self, task_id: str) -> list[float]:
        """All accuracy measurements for ``task_id``, in order."""
        return list(self._history.get(task_id, []))

    def task_ids(self) -> list[str]:
        """Task ids with at least one measurement, in insertion order."""
        return list(self._history)

    def forgetting(self, task_id: str) -> float:
        """``best_so_far - latest``; ``0.0`` when there is no prior best."""
        history = self._history.get(task_id, [])
        if len(history) < 2:
            return 0.0
        return float(max(history[:-1]) - history[-1])

    def average_forgetting(self) -> float:
        """Mean forgetting over tasks with at least two measurements."""
        forgettings = [self.forgetting(task) for task in self._history]
        if not forgettings:
            return 0.0
        return sum(forgettings) / len(forgettings)

    def summary(self) -> dict[str, Any]:
        """Snapshot of histories and derived metrics."""
        return {
            "tasks": {task: list(history) for task, history in self._history.items()},
            "average_forgetting": self.average_forgetting(),
            "num_tasks": len(self._history),
            "total_measurements": sum(len(h) for h in self._history.values()),
        }


__all__ = ["ContinualEvaluator"]
