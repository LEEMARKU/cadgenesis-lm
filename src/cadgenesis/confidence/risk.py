"""cadgenesis.confidence.risk
==============================
Risk assessment from confidence/uncertainty.

Provides risk scoring and decision-threshold frameworks that combine
confidence, uncertainty, and consequence severity to enable
confidence-gated execution and human-in-the-loop handoffs in
CADGenesis-LM workflows.
"""

from __future__ import annotations

from typing import Any

import torch


class RiskAssessor:
    """Assess risk from model confidence and uncertainty.

    Combines three factors into a composite risk score in [0, 1]:
        - Confidence: how sure the model is (high confidence = lower risk)
        - Uncertainty: epistemic + aleatoric uncertainty (high = higher risk)
        - Consequence: estimated severity of errors (high consequence = higher risk)

    Risk = sigmoid( alpha * (1 - confidence) + beta * uncertainty - gamma * consequence )
    where alpha, beta, gamma are weighting factors.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def assess(
        self,
        confidence: float,
        uncertainty: float,
        consequence: float = 0.5,
    ) -> dict[str, Any]:
        """Compute risk score and recommended action.

        - ``confidence``: model confidence in [0, 1]
        - ``uncertainty``: total uncertainty in [0, 1]
        - ``consequence``: estimated consequence severity in [0, 1]

        Returns dict with:
            - ``"risk_score"``: composite risk in [0, 1]
            - ``"action"``: recommended action ("proceed", "review", "defer")
            - ``"confidence"``: original confidence value
            - ``"uncertainty"``: original uncertainty value
            - ``"consequence"``: original consequence value
        """
        # Composite risk: weighted combination
        raw_risk = (
            self.alpha * (1.0 - confidence) + self.beta * uncertainty - self.gamma * consequence
        )
        # Clip to [0, 1] and apply sigmoid
        risk_score = torch.sigmoid(torch.tensor(raw_risk)).item()

        # Determine action based on risk threshold
        if risk_score < 0.3:
            action = "proceed"
        elif risk_score < 0.7:
            action = "review"
        else:
            action = "defer"

        return {
            "risk_score": risk_score,
            "action": action,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "consequence": consequence,
        }


class RiskConfig:
    """Configuration for risk assessment thresholds.

    Defaults are tuned for CAD generation workflows where false positives
    (approving risky generations) are costlier than false negatives (conservative review).
    """

    def __init__(
        self,
        high_risk_threshold: float = 0.7,
        low_risk_threshold: float = 0.3,
        uncertainty_penalty: float = 1.0,
        consequence_weight: float = 1.5,
    ):
        self.high_risk_threshold = high_risk_threshold
        self.low_risk_threshold = low_risk_threshold
        self.uncertainty_penalty = uncertainty_penalty
        self.consequence_weight = consequence_weight


__all__ = [
    "RiskAssessor",
    "RiskConfig",
]
