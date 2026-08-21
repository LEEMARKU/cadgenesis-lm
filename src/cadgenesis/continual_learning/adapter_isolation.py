"""cadgenesis.continual_learning.adapter_isolation
==============================================
Task-isolated adapters to prevent cross-task interference.

During continual training only task-specific parameters (adapters, LoRA
matrices, ...) may move.  The registry performs pure parameter-name
bookkeeping: it records which parameters belong to a task and freezes /
unfreezes them via ``requires_grad``, restoring the previous state on release.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch.nn as nn


@dataclass
class TaskIsolation:
    """Isolation mask for one task.

    ``param_names`` are the parameters that stay trainable while the task is
    active; ``frozen`` are the parameters that are frozen during the task.
    """

    task_id: str
    param_names: list[str] = field(default_factory=list)
    frozen: set[str] = field(default_factory=set)


class TaskAdapterRegistry:
    """Per-task parameter masks over a shared model.

    Registration matches parameter names against ``key_substrings`` (e.g.
    ``["lora", "adapter"]``); no weights are copied or stored.
    """

    def __init__(self) -> None:
        self.tasks: dict[str, TaskIsolation] = {}
        self._saved_flags: dict[str, dict[str, bool]] = {}

    def register_task(
        self,
        task_id: str,
        model: nn.Module,
        key_substrings: Sequence[str],
    ) -> TaskIsolation:
        """Capture the parameters belonging to ``task_id`` into a mask.

        A parameter is *isolated* when its name contains any of
        ``key_substrings``.  All other parameters are recorded as frozen for
        the duration of the task.
        """
        if task_id in self.tasks:
            raise ValueError(f"task {task_id!r} is already registered")
        all_names = [name for name, _ in model.named_parameters()]
        isolated = [name for name in all_names if any(k in name for k in key_substrings)]
        task = TaskIsolation(
            task_id=task_id,
            param_names=isolated,
            frozen=set(all_names) - set(isolated),
        )
        self.tasks[task_id] = task
        return task

    def isolate(self, task_id: str, model: nn.Module) -> None:
        """Freeze non-isolated parameters; unfreeze the isolated ones.

        The pre-isolation ``requires_grad`` flags are remembered on first
        isolation so :meth:`release` can restore them exactly.
        """
        task = self._require(task_id)
        if task_id not in self._saved_flags:
            self._saved_flags[task_id] = {
                name: param.requires_grad for name, param in model.named_parameters()
            }
        isolated = set(task.param_names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in isolated)

    def release(self, task_id: str, model: nn.Module) -> None:
        """Restore the ``requires_grad`` flags captured before isolation.

        The restored state is the baseline recorded when :meth:`isolate` was
        first called for ``task_id`` (i.e. the flags in effect before that
        isolation).  Masks of other concurrently-active tasks are then
        re-applied in isolation order, so releasing one task never unmasks
        a still-active task.
        """
        self._require(task_id)
        saved = self._saved_flags.pop(task_id, None)
        if saved is None:
            return
        for name, param in model.named_parameters():
            if name in saved:
                param.requires_grad_(saved[name])
        for active_id in self._saved_flags:
            active = self.tasks[active_id]
            isolated = set(active.param_names)
            for name, param in model.named_parameters():
                param.requires_grad_(name in isolated)

    def is_isolated(self, task_id: str) -> bool:
        """True when ``task_id`` is registered and currently isolated."""
        return task_id in self._saved_flags

    def _require(self, task_id: str) -> TaskIsolation:
        task = self.tasks.get(task_id)
        if task is None:
            raise ValueError(f"task {task_id!r} is not registered")
        return task


__all__ = ["TaskAdapterRegistry", "TaskIsolation"]
