"""cadgenesis.continual_learning.continual_trainer
==============================================
Continual training loop over sequential task curricula.

A minimal, dependency-free trainer: standard torch loop (``train()``,
``zero_grad``, ``backward``, ``step``) with optional regularization terms
(EWC penalties, knowledge-anchor losses, ...) added to the task loss.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn as nn

from cadgenesis.continual_learning.evaluator import ContinualEvaluator


def _default_optimizer_factory(params: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
    return torch.optim.Adam(params, lr=1e-2)


class ContinualTrainer:
    """Trains a shared model across sequential tasks.

    ``regularizers`` are callables mapping the model to a loss tensor
    (e.g. :meth:`~cadgenesis.continual_learning.ewc.EWC.penalty` or
    :meth:`~cadgenesis.continual_learning.knowledge_anchor.KnowledgeAnchor.anchor_loss`);
    their outputs are summed into the task loss each step.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_factory: Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer] | None = None,
        device: str = "cpu",
        regularizers: list[Callable[[nn.Module], torch.Tensor]] | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.optimizer_factory = optimizer_factory or _default_optimizer_factory
        self.regularizers = regularizers or []
        self.model.to(self.device)
        self.optimizer = self.optimizer_factory(self.model.parameters())
        self._tasks: list[str] = []

    def train_task(
        self,
        task_id: str,
        train_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
        epochs: int = 1,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Train ``epochs`` passes over ``train_batches``.

        The batch iterable is materialized once so the final accuracy can be
        measured on the same data.  Returns ``{"loss_steps": [...], "final_acc": float}``
        where ``final_acc`` is the accuracy on the training batches after the
        last epoch.
        """
        batches = list(train_batches)
        criterion = loss_fn or nn.CrossEntropyLoss()
        self.model.train()
        loss_steps: list[float] = []
        for _ in range(epochs):
            for inputs, targets in batches:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(inputs)
                loss = criterion(logits, targets)
                for regularizer in self.regularizers:
                    loss = loss + regularizer(self.model)
                loss.backward()
                self.optimizer.step()
                loss_steps.append(float(loss.item()))
        self._tasks.append(task_id)
        return {"loss_steps": loss_steps, "final_acc": self.evaluate(batches)}

    def fit_and_evaluate(
        self,
        task_id: str,
        train_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
        eval_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
        evaluator: ContinualEvaluator,
        epochs: int = 1,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Train a task, score it on ``eval_batches`` and record the accuracy.

        Returns ``{"loss_steps": [...], "final_acc": float, "eval_acc": float}``;
        the evaluation accuracy is recorded into ``evaluator`` under
        ``task_id`` (usable for :meth:`ContinualEvaluator.forgetting`).
        """
        result = self.train_task(task_id, train_batches, epochs=epochs, loss_fn=loss_fn)
        eval_batch_list = list(eval_batches)
        eval_acc = self.evaluate(eval_batch_list)
        evaluator.record_task_acc(task_id, eval_acc)
        return {
            "loss_steps": result["loss_steps"],
            "final_acc": result["final_acc"],
            "eval_acc": eval_acc,
        }

    @torch.no_grad()
    def evaluate(self, batches: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Mean argmax accuracy over ``batches`` (model switched to eval mode)."""
        self.model.eval()
        correct = 0
        total = 0
        for inputs, targets in batches:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            predictions = self.model(inputs).argmax(dim=-1)
            correct += int((predictions == targets).sum().item())
            total += targets.numel()
        return correct / max(1, total)

    @property
    def tasks(self) -> list[str]:
        """Task ids trained so far, in order."""
        return list(self._tasks)


__all__ = ["ContinualTrainer"]
