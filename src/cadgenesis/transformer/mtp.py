"""
cadgenesis.transformer.mtp
==========================
Multi-Token Prediction (MTP) heads in the style of DeepSeek-V3, simplified to
be dependency-free.

Each depth ``d`` predicts the token ``d`` positions ahead of the current one
using a depth-specific residual branch on top of the previous depth's hidden
state (teacher forcing with the true tokens). Logits are weight-tied with the
output vocabulary embedding: ``logits = hidden @ embed.weight.T``.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from cadgenesis.transformer.losses import MaskedCrossEntropyLoss
from cadgenesis.transformer.transformer_block import RMSNorm

__all__ = ["MultiTokenPredictionHead", "mtp_loss"]


class MultiTokenPredictionHead(nn.Module):
    """Predict the next ``mtp_depth`` tokens at each position via depth-specific
    residual branches (DeepSeek-V3 style MTP module, dependency-free).
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        mtp_depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if mtp_depth < 1:
            raise ValueError(f"mtp_depth must be >= 1, got {mtp_depth}")
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.mtp_depth = mtp_depth
        self.norms = nn.ModuleDict({str(d): RMSNorm(d_model) for d in range(1, mtp_depth + 1)})
        self.branches = nn.ModuleDict(
            {
                str(d): nn.Sequential(
                    RMSNorm(d_model),
                    nn.Linear(d_model, d_model),
                    nn.Dropout(dropout),
                )
                for d in range(1, mtp_depth + 1)
            }
        )

    def forward(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        embed: nn.Embedding,
    ) -> list[torch.Tensor]:
        """
        Parameters
        ----------
        hidden : (B, T, d_model) final-layer hidden states.
        targets : (B, T) true token ids at each position (teacher forcing).
        embed : the model's output-vocabulary Embedding module; logits are
            weight-tied as ``hidden @ embed.weight.T``.

        Returns
        -------
        logits_list : list of ``mtp_depth`` tensors
            ``logits_list[d-1]`` has shape ``(B, T-d, V)`` and predicts
            ``targets[:, d:]``.
        """
        _, T, _ = hidden.shape
        h_prev = hidden
        logits_list: list[torch.Tensor] = []
        for d in range(1, self.mtp_depth + 1):
            if d == 1:
                h_d = self.norms[str(d)](hidden + self.branches[str(d)](hidden))
            else:
                h_prev_sliced = h_prev[:, : T - d + 1]
                embed_slice = embed(targets[:, d - 1 :])
                h_d = self.norms[str(d)](embed_slice + self.branches[str(d)](h_prev_sliced))
            logits_d = h_d @ embed.weight.T
            logits_list.append(logits_d[:, : T - d])
            h_prev = h_d
        return logits_list


def mtp_loss(
    logits_list: list[torch.Tensor],
    targets: torch.Tensor,
    mtp_depth: int,
    pad_id: int = 0,
    label_smoothing: float = 0.0,
    weights: list[float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sum of per-depth masked cross-entropy losses, equally weighted by default.

    Returns ``(total, breakdown)`` where ``breakdown`` maps ``"mtp_d1"``,
    ``"mtp_d2"``, ... and ``"mtp_total"`` to float values.
    """
    if weights is None:
        weights = [1.0 / mtp_depth] * mtp_depth
    if len(weights) != mtp_depth:
        raise ValueError(f"expected {mtp_depth} weights, got {len(weights)}")
    criterion = MaskedCrossEntropyLoss(pad_id, label_smoothing)
    per_depth: list[torch.Tensor] = []
    breakdown: dict[str, float] = {}
    for d in range(1, mtp_depth + 1):
        loss_d = criterion(logits_list[d - 1], targets[:, d:])
        per_depth.append(loss_d)
        breakdown[f"mtp_d{d}"] = float(loss_d.item())
    total = cast(
        torch.Tensor, sum(w * loss_d for w, loss_d in zip(weights, per_depth, strict=True))
    )
    breakdown["mtp_total"] = float(total.item())
    return total, breakdown
