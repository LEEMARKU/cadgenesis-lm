"""
cadgenesis.confidence.confidence_engine
======================================
Confidence-Aware Intelligence & Uncertainty Calibration for CADGenesis-LM v2.0:
- Calibrated token-level & sequence-level confidence scoring
- Entropy-based uncertainty estimation
- Confidence-gated execution pipeline
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class ConfidenceEngine:
    """
    Computes calibrated confidence metrics from model logits and uncertainty heads.
    """

    @staticmethod
    def compute_sequence_confidence(
        logits: torch.Tensor,
        confidence_head_output: torch.Tensor,
    ) -> tuple[float, float]:
        """
        logits: (1, T, V)
        confidence_head_output: (1, T, 1)

        Returns: (confidence_score: float [0, 1], entropy_uncertainty: float)
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean().item()

        head_score = torch.sigmoid(confidence_head_output).mean().item()
        calibrated_confidence = (1.0 / (1.0 + entropy)) * 0.5 + head_score * 0.5
        return min(1.0, max(0.0, calibrated_confidence)), entropy
