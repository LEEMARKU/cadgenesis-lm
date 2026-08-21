"""cadgenesis.evaluation.reasoning_metrics
=======================================
Reasoning / symbolic evaluation metrics (v6.0, Pillar 7).

Measures the quality of the neuro-symbolic stack:

* :func:`reasoning_accuracy` — agreement between predicted and expected outcomes;
* :func:`symbolic_consistency` — ratio of validated conclusions;
* :func:`rule_utilization` — fraction of the rule set that fired;
* :func:`engineering_correctness` — standards compliance pass rate;
* :func:`manufacturing_correctness` — DFM checks passed;
* :func:`constraint_reasoning` — satisfiable constraint batches;
* :func:`topology_reasoning` — valid topology analyses;
* :func:`run_reasoning_benchmark` — aggregates everything into one report.
"""

from __future__ import annotations

from typing import Any


def reasoning_accuracy(
    predicted: list[bool],
    expected: list[bool],
) -> float:
    """Fraction of predictions that match the expected outcomes."""
    if not predicted:
        return 0.0
    if len(predicted) != len(expected):
        raise ValueError("predicted and expected must have equal length")
    if not expected:
        return 0.0
    matches = sum(1 for p, e in zip(predicted, expected, strict=True) if bool(p) == bool(e))
    return matches / len(predicted)


def symbolic_consistency(
    validated: int,
    total: int,
) -> float:
    """Fraction of conclusions that passed symbolic verification."""
    if total <= 0:
        return 0.0
    return validated / total


def rule_utilization(fired: int, available: int) -> float:
    """Fraction of the rule set that actually fired."""
    if available <= 0:
        return 0.0
    return fired / available


def engineering_correctness(
    compliance_results: list[tuple[str, bool]],
) -> float:
    """Pass rate of engineering-standard compliance checks."""
    if not compliance_results:
        return 0.0
    passed = sum(1 for _, ok in compliance_results if ok)
    return passed / len(compliance_results)


def manufacturing_correctness(
    checks: list[tuple[str, bool]],
) -> float:
    """Pass rate of DFM rule checks."""
    if not checks:
        return 0.0
    passed = sum(1 for _, ok in checks if ok)
    return passed / len(checks)


def constraint_reasoning(
    solutions: list[tuple[bool, float]],
) -> float:
    """Fraction of constraint batches solved within tolerance."""
    if not solutions:
        return 0.0
    feasible = sum(1 for ok, _ in solutions if ok)
    return feasible / len(solutions)


def topology_reasoning(
    analyses: list[tuple[bool, int]],
) -> float:
    """Fraction of topology analyses judged valid.

    ``analyses`` is a list of ``(valid, faces)`` pairs.
    """
    if not analyses:
        return 0.0
    valid = sum(1 for ok, _ in analyses if ok)
    return valid / len(analyses)


def run_reasoning_benchmark(
    predictions: list[tuple[bool, bool]] | None = None,
    conclusions: tuple[int, int] = (0, 0),
    rule_usage: tuple[int, int] = (0, 0),
    compliance: list[tuple[str, bool]] | None = None,
    dfm_checks: list[tuple[str, bool]] | None = None,
    constraint_batches: list[tuple[bool, float]] | None = None,
    topology_samples: list[tuple[bool, int]] | None = None,
) -> dict[str, Any]:
    """Aggregate the standard reasoning metrics into one report."""
    predicted = [p for p, _ in (predictions or [])]
    expected = [e for _, e in (predictions or [])]
    report: dict[str, Any] = {
        "reasoning_accuracy": reasoning_accuracy(predicted, expected),
        "symbolic_consistency": symbolic_consistency(*conclusions),
        "rule_utilization": rule_utilization(*rule_usage),
        "engineering_correctness": engineering_correctness(compliance or []),
        "manufacturing_correctness": manufacturing_correctness(dfm_checks or []),
        "constraint_reasoning": constraint_reasoning(constraint_batches or []),
        "topology_reasoning": topology_reasoning(topology_samples or []),
        "checks": {
            "predictions": len(predicted),
            "conclusions": conclusions[1],
            "rules": rule_usage[1],
            "compliance": len(compliance or []),
            "dfm_checks": len(dfm_checks or []),
            "constraint_batches": len(constraint_batches or []),
            "topology_samples": len(topology_samples or []),
        },
    }
    return report


__all__ = [
    "constraint_reasoning",
    "engineering_correctness",
    "manufacturing_correctness",
    "reasoning_accuracy",
    "rule_utilization",
    "run_reasoning_benchmark",
    "symbolic_consistency",
    "topology_reasoning",
]
