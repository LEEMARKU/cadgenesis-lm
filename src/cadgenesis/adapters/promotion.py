"""cadgenesis.adapters.promotion
=============================
Adapter promotion to shared weights.

Evaluates adapter metrics against promotion criteria (matching the manager's
``accuracy > 0.85`` / ``stability > 0.90`` thresholds by default) and returns
decisions; never mutates the manager directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from cadgenesis.adapters.manager import AdapterMetadata

PROMOTED_STATUS = "promoted"


@dataclass(frozen=True)
class PromotionCriteria:
    """Thresholds an adapter must satisfy to be promoted."""

    min_accuracy: float = 0.85
    min_stability: float = 0.90
    max_drift: float = 0.10
    min_samples: int = 1


@dataclass(frozen=True)
class PromotionDecision:
    """Result of an :meth:`AdapterPromotion.evaluate` run."""

    approved: bool
    reasons: list[str]
    score: float


class AdapterPromotion:
    """Promotion gate: evaluates metrics against criteria and reports why."""

    def evaluate(
        self,
        metadata: AdapterMetadata,
        metrics: dict[str, float],
        criteria: PromotionCriteria | None = None,
    ) -> PromotionDecision:
        """Score ``metrics`` (accuracy, stability, optional drift, samples)."""
        criteria = criteria if criteria is not None else PromotionCriteria()
        accuracy = metrics.get("accuracy", metadata.accuracy_score)
        stability = metrics.get("stability", metadata.stability_score)
        drift = metrics.get("drift")
        samples = int(metrics.get("samples", 0))

        reasons: list[str] = []
        passed: list[bool] = []
        if accuracy >= criteria.min_accuracy:
            reasons.append(f"accuracy {accuracy:.3f} >= {criteria.min_accuracy:.3f}")
            passed.append(True)
        else:
            reasons.append(f"accuracy {accuracy:.3f} < required {criteria.min_accuracy:.3f}")
            passed.append(False)
        if stability >= criteria.min_stability:
            reasons.append(f"stability {stability:.3f} >= {criteria.min_stability:.3f}")
            passed.append(True)
        else:
            reasons.append(f"stability {stability:.3f} < required {criteria.min_stability:.3f}")
            passed.append(False)
        if drift is not None:
            if drift <= criteria.max_drift:
                reasons.append(f"drift {drift:.3f} <= {criteria.max_drift:.3f}")
                passed.append(True)
            else:
                reasons.append(f"drift {drift:.3f} > allowed {criteria.max_drift:.3f}")
                passed.append(False)
        if samples >= criteria.min_samples:
            reasons.append(f"samples {samples} >= {criteria.min_samples}")
            passed.append(True)
        else:
            reasons.append(f"samples {samples} < required {criteria.min_samples}")
            passed.append(False)

        score = 0.5 * accuracy + 0.3 * stability
        if drift is not None:
            score += 0.2 * (1.0 - drift)
        return PromotionDecision(approved=all(passed), reasons=reasons, score=score)

    def promote(
        self,
        metadata: AdapterMetadata,
        metrics: dict[str, float],
        criteria: PromotionCriteria | None = None,
    ) -> str:
        """Apply the decision to ``metadata`` and return the new status string."""
        decision = self.evaluate(metadata, metrics, criteria)
        if decision.approved:
            metadata.status = PROMOTED_STATUS
        return metadata.status
