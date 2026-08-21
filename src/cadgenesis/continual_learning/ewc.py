"""cadgenesis.continual_learning.ewc
=================================
Elastic Weight Consolidation regularization for catastrophic forgetting.

EWC penalizes movement of model parameters away from a set of *anchors* taken
after previous tasks.  The penalty is a quadratic form weighted by the diagonal
Fisher information estimate::

    L_ewc = 0.5 * lambda * sum_i F_i * (theta_i - theta_star_i)^2

``F_i`` is the diagonal of the Fisher information for parameter ``i``
(approximated as the mean squared gradient over a sample of the previous
task's data), and ``theta_star_i`` is the anchored parameter value.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn as nn


class EWC:
    """Elastic weight consolidation: quadratic penalty around frozen anchors.

    Pure torch, no ``nn.Module`` requirement: :meth:`penalty` returns a tensor
    that can be added directly to the task loss inside ``loss.backward()``.
    """

    def __init__(self, lambda_: float = 1000.0) -> None:
        self.lambda_ = lambda_
        self.anchors: dict[str, torch.Tensor] = {}
        self.fishers: dict[str, torch.Tensor] = {}
        self._fisher_sums: dict[str, torch.Tensor] = {}
        self._steps: int = 0

    def register_fisher(
        self,
        model: nn.Module,
        dataloader_or_samples: Iterable[tuple[Any, Any]],
        loss_fn: Callable[[Any, Any], torch.Tensor],
        n_samples: int = 200,
    ) -> None:
        """Estimate the diagonal Fisher information and re-anchor.

        Iterates at most ``n_samples`` batches (each ``(inputs, targets)``
        pair drawn from ``dataloader_or_samples``), computes the task loss and
        back-propagates, accumulating ``param.grad.pow(2)`` per step.  Gradients
        that are ``None`` (unused parameters) are skipped.  The anchors are
        refreshed to the current weights at the end of the pass.
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        seen_steps = 0
        for inputs, targets in dataloader_or_samples:
            if seen_steps >= n_samples:
                break
            model.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                running = self._fisher_sums.get(name)
                if running is None:
                    running = torch.zeros_like(grad)
                self._fisher_sums[name] = running + grad.pow(2)
            seen_steps += 1
        if seen_steps == 0:
            raise ValueError("register_fisher saw no batches to estimate the Fisher")
        self._steps += seen_steps
        self.fishers = {
            name: (total / self._steps).clone() for name, total in self._fisher_sums.items()
        }
        self.anchors = {name: param.detach().clone() for name, param in model.named_parameters()}

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Return the EWC penalty ``0.5 * lambda * sum(F * (theta - theta*)^2)``.

        Only parameters present in both :attr:`fishers` and :attr:`anchors`
        contribute.  The returned tensor carries a gradient path back to the
        current parameters, so it can be summed into the training loss.
        """
        terms: list[torch.Tensor] = []
        for name, param in model.named_parameters():
            anchor = self.anchors.get(name)
            fisher = self.fishers.get(name)
            if anchor is None or fisher is None:
                continue
            diff = param - anchor.to(param.device)
            terms.append((fisher.to(param.device) * diff.pow(2)).sum())
        if not terms:
            return torch.tensor(0.0)
        return 0.5 * self.lambda_ * torch.stack(terms).sum()

    def ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """Alias for :meth:`penalty` (EWC regularization term of the loss)."""
        return self.penalty(model)

    def reset(self) -> None:
        """Drop all Fisher estimates, anchors and accumulation state."""
        self.anchors.clear()
        self.fishers.clear()
        self._fisher_sums.clear()
        self._steps = 0


__all__ = ["EWC"]
