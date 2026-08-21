"""cadgenesis.transformer.losses
=============================
Training losses for CADGenesis-LM v6.0: masked language-modeling cross entropy,
a confidence-regularisation term, and a combined ``CADSequenceLoss`` used by
the trainer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCrossEntropyLoss(nn.Module):
    """Cross-entropy over logits ignoring positions labelled with ``pad_id``."""

    def __init__(self, pad_id: int = 0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """logits: (B, T, V); targets: (B, T); mask: optional (B, T) bool."""
        if mask is None:
            mask = targets != self.pad_id
        if mask.dtype != torch.bool:
            mask = mask.bool()
        logits_flat = logits[mask]
        targets_flat = targets[mask]
        if logits_flat.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction="mean",
            label_smoothing=self.label_smoothing,
        )


class ConfidenceLoss(nn.Module):
    """Binary cross-entropy between predicted confidence and a target scalar.

    Used to teach the confidence head the model's *actual* correctness.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"invalid reduction {reduction!r}")
        self.reduction = reduction

    def forward(
        self,
        confidence_logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """confidence_logits: (B, T, 1); targets: (B, T) in [0, 1]."""
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.bool()
        logits = confidence_logits.squeeze(-1)
        if mask is not None:
            logits = logits[mask]
            targets = targets[mask]
        if logits.numel() == 0:
            return torch.tensor(0.0, device=confidence_logits.device, requires_grad=True)
        return F.binary_cross_entropy_with_logits(logits, targets, reduction=self.reduction)


class CADSequenceLoss(nn.Module):
    """Combined CAD sequence loss: masked CE + optional confidence term.

    When ``moe_aux_scale > 0`` and an ``aux_loss`` callable (e.g. the MoE
    load-balancing loss) is supplied, its value is added to the total.
    """

    def __init__(
        self,
        pad_id: int = 0,
        label_smoothing: float = 0.0,
        confidence_weight: float = 0.1,
        moe_aux_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if confidence_weight < 0 or moe_aux_scale < 0:
            raise ValueError("loss weights must be non-negative")
        self.ce_loss = MaskedCrossEntropyLoss(pad_id, label_smoothing)
        self.confidence_loss = ConfidenceLoss()
        self.confidence_weight = confidence_weight
        self.moe_aux_scale = moe_aux_scale

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        confidence_logits: torch.Tensor | None = None,
        target_confidence: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        aux_loss: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return ``(total_loss, breakdown_dict)``.

        ``target_confidence`` should be 1.0 for tokens the model predicted
        correctly and 0.0 otherwise; when omitted the confidence term is
        skipped.
        """
        ce = self.ce_loss(logits, targets, mask=mask)
        breakdown: dict[str, float] = {"ce": float(ce.item())}

        confidence_term = torch.tensor(0.0, device=logits.device)
        if confidence_logits is not None and target_confidence is not None:
            conf_mask = mask if mask is not None else targets != 0
            confidence_term = self.confidence_loss(
                confidence_logits, target_confidence, mask=conf_mask
            )
            breakdown["confidence"] = float(confidence_term.item())

        aux_term = torch.tensor(0.0, device=logits.device)
        if aux_loss is not None:
            aux_term = self.moe_aux_scale * aux_loss
            breakdown["moe_aux"] = float(aux_term.item())

        total = ce + self.confidence_weight * confidence_term + aux_term
        breakdown["total"] = float(total.item())
        return total, breakdown
