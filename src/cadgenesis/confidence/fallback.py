"""
cadgenesis.confidence.fallback
==============================
Low-confidence fallback strategies for CADGenesis-LM v2.0.

When model confidence drops below a threshold, predefined fallback
strategies are triggered to maintain safe operation without requiring
human intervention in every case.
"""

from __future__ import annotations

from typing import TypedDict


class FallbackStrategy:
    """Enum-like class for fallback strategies."""

    RETRY = "retry"
    REGENERATE = "regenerate"
    SAFE_TEMPLATE = "safe_template"
    DEFER_TO_HUMAN = "defer_to_human"
    USE_DEFAULT = "use_default"
    ABSTAIN = "abstain"


class FallbackDecision(TypedDict):
    """Result of a fallback strategy decision."""

    strategy: str
    reason: str
    confidence: float


class FallbackPolicy:
    """
    Policy for determining when and which fallback strategy to apply.

    The policy evaluates the current confidence level, uncertainty,
    and consequence severity to select an appropriate strategy.
    """

    def __init__(
        self,
        low_confidence: float = 0.5,
        high_uncertainty: float = 0.7,
        critical_consequence: float = 0.9,
    ):
        self.low_confidence = low_confidence
        self.high_uncertainty = high_uncertainty
        self.critical_consequence = critical_consequence

    def decide(
        self,
        confidence: float,
        uncertainty: float,
        consequence: float = 0.5,
    ) -> FallbackDecision:
        """
        Determine the appropriate fallback strategy.

        - ``confidence``: current model confidence in [0, 1]
        - ``uncertainty``: current model uncertainty in [0, 1]
        - ``consequence``: estimated consequence severity in [0, 1]

        Returns: ``FallbackDecision`` with strategy, reason, and original confidence.
        """
        # High confidence: no fallback needed
        if confidence > self.low_confidence and uncertainty < self.high_uncertainty:
            return {
                "strategy": "retry",
                "reason": "High confidence, low uncertainty",
                "confidence": confidence,
            }

        # Low confidence + low uncertainty: retry
        if confidence < self.low_confidence and uncertainty < self.high_uncertainty:
            return {
                "strategy": "retry",
                "reason": "Low confidence, retry generation",
                "confidence": confidence,
            }

        # Low confidence + high uncertainty: regenerate with fresh prompt
        if confidence < self.low_confidence and uncertainty >= self.high_uncertainty:
            return {
                "strategy": "regenerate",
                "reason": "Low confidence + high uncertainty, regenerate",
                "confidence": confidence,
            }

        # High uncertainty + critical consequence: defer to human
        if uncertainty >= self.high_uncertainty and consequence >= self.critical_consequence:
            return {
                "strategy": "defer_to_human",
                "reason": "High uncertainty + critical consequence, human review needed",
                "confidence": confidence,
            }

        # Default: use safe template or default configuration
        return {
            "strategy": "use_default",
            "reason": "Default fallback applied",
            "confidence": confidence,
        }


class AbstentionPolicy:
    """Threshold-based abstention: withhold low-confidence outputs.

    A model may return ``abstain`` when its calibrated confidence falls
    below ``threshold`` (or when uncertainty exceeds ``max_uncertainty``),
    so downstream systems can defer, retry, or re-prompt instead of acting
    on an unreliable prediction.
    """

    def __init__(
        self,
        threshold: float = 0.6,
        max_uncertainty: float | None = None,
    ):
        self.threshold = threshold
        self.max_uncertainty = max_uncertainty

    def should_abstain(
        self,
        confidence: float,
        uncertainty: float | None = None,
    ) -> bool:
        """True when the prediction should be withheld."""
        if confidence < self.threshold:
            return True
        if self.max_uncertainty is not None and uncertainty is not None:
            if uncertainty > self.max_uncertainty:
                return True
        return False

    def decide(
        self,
        confidence: float,
        uncertainty: float | None = None,
    ) -> FallbackDecision:
        """Return a fallback decision; ``abstain`` when below threshold."""
        if self.should_abstain(confidence, uncertainty):
            reason = "confidence below threshold" if confidence < self.threshold else "uncertainty above limit"
            return {
                "strategy": FallbackStrategy.ABSTAIN,
                "reason": reason,
                "confidence": confidence,
            }
        return {
            "strategy": FallbackStrategy.RETRY,
            "reason": "confidence above threshold",
            "confidence": confidence,
        }

    def abstention_rate(self, confidences: list[float]) -> float:
        """Fraction of predictions withheld under this policy."""
        if not confidences:
            return 0.0
        return sum(1.0 for c in confidences if self.should_abstain(c)) / len(confidences)

    def selective_accuracy(
        self,
        confidences: list[float],
        correct: list[bool],
    ) -> float:
        """Accuracy on the predictions that were NOT abstained.

        Requires one ``correct`` flag per confidence; pairs are zipped.
        """
        accepted = [
            (c, ok)
            for c, ok in zip(confidences, correct, strict=False)
            if not self.should_abstain(c)
        ]
        if not accepted:
            return 0.0
        return sum(1.0 for _, ok in accepted if ok) / len(accepted)


__all__ = [
    "AbstentionPolicy",
    "FallbackDecision",
    "FallbackPolicy",
    "FallbackStrategy",
]
