"""
cadgenesis.confidence.confidence
================================
Confidence scoring engine for CADGenesis-LM v2.0.

Provides calibrated confidence scores from model logits, combining
temperature-scaled probabilities with entropy-based uncertainty estimation.
Exposes a simple API for sequence-level and token-level confidence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .calibration import ConfidenceCalibrator


class ConfidenceEngine:
    """
    High-level confidence scoring engine.

    Combines calibrated probabilities (via temperature scaling)
    with uncertainty estimation (entropy-based) to produce confidence
    scores suitable for confidence-gated execution.

    Attributes
    ----------
    calibrator : ConfidenceCalibrator | None
        Fitted calibrator if :meth:`fit` has been called, otherwise ``None``.
    """

    def __init__(self, calibrator: ConfidenceCalibrator | None = None):
        self.calibrator = calibrator

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, **fit_kwargs) -> None:
        """
        Fit the calibrator on ``(logits, labels)``.

        - ``logits``: (N, C) or (B, T, C) — pre-softmax scores.
        - ``labels``: (N,) or (B, T) — class indices.
        - ``fit_kwargs``: passed to :class:`ConfidenceCalibrator.fit`.
        """
        self.calibrator = ConfidenceCalibrator()
        self.calibrator.fit(logits, labels, **fit_kwargs)

    def compute_sequence_confidence(self, logits: torch.Tensor) -> tuple[float, float]:
        """
        Compute calibrated confidence and entropy uncertainty for a sequence.

        - ``logits``: (1, T, V) — batch of logits for one sequence.

        Returns: ``(confidence_score: float [0, 1], entropy_uncertainty: float)``
        """
        if self.calibrator is None:
            raise RuntimeError("Calibrator not fitted yet.")
        self.calibrator.eval()
        with torch.no_grad():
            _probs = F.softmax(logits, dim=-1)
            calibrated = self.calibrator.calibrate(logits)
            confidence = calibrated.amax(dim=-1).mean().item()
        # Entropy of calibrated probabilities
        ent = (-calibrated * torch.log(calibrated + 1e-8)).mean(dim=-1).mean().item()
        return confidence, ent

    def compute_token_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token confidence scores.

        - ``logits``: (B, T, V) — batch of logits.

        Returns: ``confidence``: (B, T) — confidence in [0, 1] per token.
        """
        if self.calibrator is None:
            raise RuntimeError("Calibrator not fitted yet.")
        self.calibrator.eval()
        with torch.no_grad():
            _probs = F.softmax(logits, dim=-1)
            calibrated = self.calibrator.calibrate(logits)
            return calibrated.amax(dim=-1).detach()

    def uncertainty(self, logits: torch.Tensor) -> dict[str, float]:
        """
        Estimate uncertainty based on calibrated probabilities.

        Returns: dict with keys ``"entropy"`` and ``"aleatoric"``.
        """
        if self.calibrator is None:
            raise RuntimeError("Calibrator not fitted yet.")
        self.calibrator.eval()
        with torch.no_grad():
            _probs = F.softmax(logits, dim=-1)
            cal = self.calibrator.calibrate(logits)
            entropy = (-cal * torch.log(cal + 1e-8)).mean(dim=-1).mean().item()
        # Aleatoric approximated by confidence margin
        margin = 1.0 - cal.amax(dim=-1).mean().item()
        return {"entropy": entropy, "aleatoric": margin}


def compute_confidence(
    logits: torch.Tensor, labels: torch.Tensor, calibrator: ConfidenceCalibrator | None = None
) -> tuple[float, float]:
    """
    Convenience function to compute confidence without instantiating
    a full :class:`ConfidenceEngine`.

    - ``logits``: (N, C) or (B, T, C) — pre-softmax scores.
    - ``labels``: (N,) or (B, T) — class indices.
    - ``calibrator``: optional pre-fitted :class:`ConfidenceCalibrator`.

    Returns: ``(confidence_score: float [0, 1], entropy_uncertainty: float)``
    """
    if calibrator is None:
        calibrator = ConfidenceCalibrator()
        calibrator.fit(logits, labels)
    engine = ConfidenceEngine(calibrator=calibrator)
    return engine.compute_sequence_confidence(logits)


def compute_sequence_confidence(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """
    Convenience function.

    - ``logits``: (N, C) or (B, T, C) — pre-softmax scores.
    - ``labels``: (N,) or (B, T) — class indices.

    Returns: ``(confidence_score: float [0, 1], entropy_uncertainty: float)``
    """
    engine = ConfidenceEngine()
    engine.fit(logits, labels)
    return engine.compute_sequence_confidence(logits)


__all__ = [
    "ConfidenceEngine",
    "compute_confidence",
    "compute_sequence_confidence",
]
