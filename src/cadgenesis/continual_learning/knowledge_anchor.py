"""cadgenesis.continual_learning.knowledge_anchor
==============================================
Knowledge anchors: stable parameter anchors across tasks.

A knowledge anchor snapshots the model parameters at a task boundary.  The
anchor loss is a normalized mean-squared-error over the anchored parameters,
weighted so each parameter contributes ``1 / numel`` (renormalized to sum to
one) — large matrices are not implicitly privileged over small ones.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class KnowledgeAnchor:
    """Detached parameter snapshot with a weighted MSE loss and restore."""

    def __init__(self) -> None:
        self.anchors: dict[str, torch.Tensor] = {}
        self._weights: dict[str, torch.Tensor] = {}

    def anchor(self, model: nn.Module) -> None:
        """Snapshot the current parameter tensors (detached clones)."""
        params = dict(model.named_parameters())
        self.anchors = {name: p.detach().clone() for name, p in params.items()}
        raw = {name: 1.0 / max(1, p.numel()) for name, p in params.items()}
        total = sum(raw.values())
        self._weights = {name: torch.tensor(w / total) for name, w in raw.items()}

    def anchor_loss(self, model: nn.Module) -> torch.Tensor:
        """Weighted MSE between current and anchored parameters.

        Each anchored parameter contributes ``weight * mean((theta - theta*)^2)``
        with ``weight = (1 / numel)`` normalized over the anchor.  Only
        parameters present in the anchor are scored.
        """
        if not self.anchors:
            return torch.tensor(0.0)
        terms: list[torch.Tensor] = []
        for name, param in model.named_parameters():
            anchor = self.anchors.get(name)
            weight = self._weights.get(name)
            if anchor is None or weight is None:
                continue
            diff = param - anchor.to(param.device)
            terms.append(weight.to(param.device) * diff.pow(2).mean())
        if not terms:
            return torch.tensor(0.0)
        return torch.stack(terms).sum()

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Copy the anchored tensors back into the model (in place)."""
        names = {name for name, _ in model.named_parameters()}
        subset = {name: self.anchors[name] for name in names if name in self.anchors}
        if subset:
            model.load_state_dict(subset, strict=False)

    def move_anchor(self, model: nn.Module) -> None:
        """Re-anchor to the current weights (alias of :meth:`anchor`)."""
        self.anchor(model)

    @property
    def is_anchored(self) -> bool:
        """True once at least one parameter has been anchored."""
        return bool(self.anchors)

    def __len__(self) -> int:
        return len(self.anchors)


__all__ = ["KnowledgeAnchor"]
