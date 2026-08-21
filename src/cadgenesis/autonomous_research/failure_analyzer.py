"""
Failure Analyzer - Detect convergence failures, instability, catastrophic forgetting,
hallucinations, engineering failures.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any

try:
    import numpy as np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - numpy ships with torch
    import statistics as _stats

    class _NpFallback:
        @staticmethod
        def var(values: list[float]) -> float:
            return _stats.variance(values) if len(values) > 1 else 0.0

        @staticmethod
        def mean(values: list[float]) -> float:
            return _stats.mean(values)

    np = _NpFallback()  # type: ignore[assignment]


class FailureType(str, Enum):
    CONVERGENCE = "convergence_failure"
    INSTABILITY = "instability"
    CATASTROPHIC_FORGETTING = "catastrophic_forgetting"
    HALLUCINATION = "hallucination"
    ENGINEERING = "engineering_failure"
    MEMORY = "memory_failure"
    NUMERICAL = "numerical_issue"


class FailureSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FailureDetection:
    """A detected failure."""

    detection_id: str
    failure_type: FailureType
    severity: FailureSeverity
    description: str
    evidence: dict[str, Any]
    suggested_fix: str
    confidence: float  # 0-1
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureReport:
    """Report of all detected failures."""

    report_id: str
    experiment_id: str
    detections: list[FailureDetection] = field(default_factory=list)
    summary: str = ""
    created_at: float = field(default_factory=time.time)


class FailureAnalyzer:
    """Analyzes experiments for various failure modes."""

    def __init__(self):
        self._reports: dict[str, FailureReport] = {}
        self._lock = RLock()

    def analyze_training_run(
        self,
        experiment_id: str,
        loss_history: list[float],
        metrics_history: dict[str, Any],
        model_outputs: list[Any] | None = None,
        ground_truth: list[Any] | None = None,
    ) -> FailureReport:
        """Analyze a training run for failures."""
        detections: list[FailureDetection] = []

        # Check convergence
        convergence_failures = self._check_convergence(loss_history)
        detections.extend(convergence_failures)

        # Check instability
        instability_failures = self._check_instability(loss_history)
        detections.extend(instability_failures)

        # Check catastrophic forgetting (if continual learning)
        if "task_accuracies" in metrics_history:
            forgetting_failures = self._check_catastrophic_forgetting(
                metrics_history["task_accuracies"]
            )
            detections.extend(forgetting_failures)

        # Check numerical issues
        numerical_failures = self._check_numerical_issues(loss_history, metrics_history)
        detections.extend(numerical_failures)

        # Check engineering failures (if outputs provided)
        if model_outputs is not None and ground_truth is not None:
            engineering_failures = self._check_engineering_failures(model_outputs, ground_truth)
            detections.extend(engineering_failures)

        report = FailureReport(
            report_id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            detections=detections,
        )

        # Generate summary
        by_severity: dict[FailureSeverity, int] = {}
        for d in detections:
            by_severity[d.severity] = by_severity.get(d.severity, 0) + 1

        report.summary = f"Detected {len(detections)} issues: " + ", ".join(
            f"{k.value}: {v}" for k, v in by_severity.items()
        )

        with self._lock:
            self._reports[report.report_id] = report

        return report

    def _check_convergence(self, loss_history: list[float]) -> list[FailureDetection]:
        detections: list[FailureDetection] = []

        if len(loss_history) < 10:
            return detections

        # Check if loss is not decreasing
        recent = loss_history[-10:]
        early = loss_history[:10] if len(loss_history) >= 20 else loss_history[:-10]

        if len(early) > 0:
            recent_avg = sum(recent) / len(recent)
            early_avg = sum(early) / len(early)

            if recent_avg >= early_avg * 0.99:  # Less than 1% improvement
                detections.append(
                    FailureDetection(
                        detection_id=str(uuid.uuid4()),
                        failure_type=FailureType.CONVERGENCE,
                        severity=FailureSeverity.MEDIUM,
                        description="Loss has plateaued with minimal improvement",
                        evidence={
                            "recent_avg": recent_avg,
                            "early_avg": early_avg,
                            "improvement": (early_avg - recent_avg) / early_avg,
                        },
                        suggested_fix=(
                            "Consider increasing learning rate, adding learning rate schedule, "
                            "or changing optimizer"
                        ),
                        confidence=0.7,
                    )
                )

        # Check for NaN/inf
        if any(
            not isinstance(v, (int, float)) or v != v or abs(v) == float("inf")
            for v in loss_history
        ):
            detections.append(
                FailureDetection(
                    detection_id=str(uuid.uuid4()),
                    failure_type=FailureType.NUMERICAL,
                    severity=FailureSeverity.CRITICAL,
                    description="NaN or Inf detected in loss",
                    evidence={"loss_history": loss_history[-5:]},
                    suggested_fix=(
                        "Add gradient clipping, reduce learning rate, check for numerical "
                        "instability in model"
                    ),
                    confidence=1.0,
                )
            )

        return detections

    def _check_instability(self, loss_history: list[float]) -> list[FailureDetection]:
        detections: list[FailureDetection] = []

        if len(loss_history) < 20:
            return detections

        # Check for high variance in recent loss
        recent = loss_history[-20:]
        variance = np.var(recent)
        mean = np.mean(recent)

        if mean > 0 and variance / (mean**2) > 0.1:  # Coefficient of variation > 10%
            detections.append(
                FailureDetection(
                    detection_id=str(uuid.uuid4()),
                    failure_type=FailureType.INSTABILITY,
                    severity=FailureSeverity.MEDIUM,
                    description="High variance in loss indicates training instability",
                    evidence={"variance": variance, "mean": mean, "cv": variance / (mean**2)},
                    suggested_fix=(
                        "Reduce learning rate, add gradient clipping, increase batch size"
                    ),
                    confidence=0.8,
                )
            )

        # Check for sudden spikes
        for i in range(1, len(recent)):
            if recent[i] > recent[i - 1] * 5:  # 5x spike
                detections.append(
                    FailureDetection(
                        detection_id=str(uuid.uuid4()),
                        failure_type=FailureType.INSTABILITY,
                        severity=FailureSeverity.HIGH,
                        description=f"Sudden loss spike at step {i}",
                        evidence={
                            "before": recent[i - 1],
                            "after": recent[i],
                            "ratio": recent[i] / recent[i - 1],
                        },
                        suggested_fix=(
                            "Add gradient clipping, check for data issues, reduce learning rate"
                        ),
                        confidence=0.9,
                    )
                )

        return detections

    def _check_catastrophic_forgetting(
        self, task_accuracies: dict[str, list[float]]
    ) -> list[FailureDetection]:
        detections = []

        for task, accuracies in task_accuracies.items():
            if len(accuracies) < 2:
                continue

            # Check if accuracy dropped significantly after learning new tasks
            peak = max(accuracies)
            current = accuracies[-1]

            if peak - current > 0.15:  # 15% drop
                detections.append(
                    FailureDetection(
                        detection_id=str(uuid.uuid4()),
                        failure_type=FailureType.CATASTROPHIC_FORGETTING,
                        severity=FailureSeverity.HIGH,
                        description=f"Catastrophic forgetting detected on task {task}",
                        evidence={
                            "peak_accuracy": peak,
                            "current_accuracy": current,
                            "drop": peak - current,
                        },
                        suggested_fix=(
                            "Enable replay buffer, use EWC, reduce learning rate for old tasks"
                        ),
                        confidence=0.85,
                    )
                )

        return detections

    def _check_numerical_issues(
        self, loss_history: list[float], metrics_history: dict[str, Any]
    ) -> list[FailureDetection]:
        detections: list[FailureDetection] = []

        # Check gradients (if available)
        if "grad_norm" in metrics_history:
            grad_norms = metrics_history["grad_norm"]
            if any(g > 100 for g in grad_norms[-10:]):
                detections.append(
                    FailureDetection(
                        detection_id=str(uuid.uuid4()),
                        failure_type=FailureType.NUMERICAL,
                        severity=FailureSeverity.HIGH,
                        description="Exploding gradients detected",
                        evidence={"max_grad_norm": max(grad_norms[-10:])},
                        suggested_fix="Add gradient clipping (max_norm=1.0), reduce learning rate",
                        confidence=0.9,
                    )
                )

        return detections

    def _check_engineering_failures(
        self, outputs: list[Any], ground_truth: list[Any]
    ) -> list[FailureDetection]:
        detections: list[FailureDetection] = []

        # Placeholder for engineering-specific checks
        # Could check: invalid CAD, impossible geometry, broken constraints, etc.

        return detections

    def get_report(self, report_id: str) -> FailureReport | None:
        with self._lock:
            return self._reports.get(report_id)

    def list_reports(self, experiment_id: str | None = None) -> list[FailureReport]:
        with self._lock:
            reports = list(self._reports.values())
            if experiment_id:
                reports = [r for r in reports if r.experiment_id == experiment_id]
            return reports
