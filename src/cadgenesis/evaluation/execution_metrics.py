"""cadgenesis.evaluation.execution_metrics
========================================
CAD execution pipeline metrics (Pillar 8).

Pure-function metrics over :class:`CADExecutionResult` objects (or anything
duck-typed with the same attributes), plus an aggregate benchmark runner.
They take plain results so any test or benchmark can plug in values.
"""

from __future__ import annotations

from typing import Any


def _results(results: list[Any]) -> list[Any]:
    return [r for r in results if r is not None]


def geometry_validity_rate(results: list[Any]) -> float:
    """Fraction of execution results with valid geometry."""
    results = _results(results)
    if not results:
        return 0.0
    return sum(1.0 for r in results if getattr(r, "is_valid_geometry", False)) / len(results)


def manufacturability_rate(results: list[Any]) -> float:
    """Fraction of execution results assessed as manufacturable."""
    results = _results(results)
    if not results:
        return 0.0
    return sum(1.0 for r in results if getattr(r, "is_manufacturable", False)) / len(results)


def confidence_agreement(results: list[Any]) -> float:
    """Mean confidence among results that pass all gating flags."""
    passing = [
        r
        for r in _results(results)
        if getattr(r, "is_valid_geometry", False) and getattr(r, "is_manufacturable", False)
    ]
    if not passing:
        return 0.0
    return sum(float(getattr(r, "confidence_score", 0.0)) for r in passing) / len(passing)


def safety_factor_pass_rate(results: list[Any], required: float = 1.5) -> float:
    """Fraction of results whose safety factor meets ``required``."""
    results = _results(results)
    if not results:
        return 0.0
    return sum(1.0 for r in results if float(getattr(r, "safety_factor", 0.0)) >= required) / len(
        results
    )


def simulation_pass_rate(results: list[Any]) -> float:
    """Fraction of results whose simulation report passed."""
    results = _results(results)
    if not results:
        return 0.0
    passed = 0
    for r in results:
        report = getattr(r, "simulation_report", None)
        if isinstance(report, dict):
            if report.get("passed") is True:
                passed += 1
        elif report is not None and getattr(report, "passed", False):
            passed += 1
    return passed / len(results)


def cost_error(estimated: list[float], actual: list[float]) -> float:
    """Mean absolute relative error between estimated and actual cost."""
    if not estimated:
        return 0.0
    pairs = zip(estimated, actual, strict=False)
    errors = [abs(est - act) / abs(act) if act else 0.0 for est, act in pairs]
    return sum(errors) / len(errors)


def repair_rate(results: list[Any]) -> float:
    """Fraction of results whose pipeline attempted an automatic repair."""
    results = _results(results)
    if not results:
        return 0.0
    repaired = 0
    for r in results:
        report = getattr(r, "repair_report", None)
        if isinstance(report, dict):
            if report.get("applied") or report.get("attempted"):
                repaired += 1
        elif report is not None:
            for key in ("applied", "attempted"):
                if getattr(report, key, False):
                    repaired += 1
                    break
    return repaired / len(results)


def _repair_attempted(result: Any) -> bool:
    report = getattr(result, "repair_report", None)
    if isinstance(report, dict):
        return bool(report.get("applied") or report.get("attempted"))
    if report is not None:
        return bool(getattr(report, "applied", False) or getattr(report, "attempted", False))
    return False


def initial_success_rate(results: list[Any]) -> float:
    """Fraction of results valid on the first attempt (no repair needed).

    A result counts as initially successful when geometry is valid, no
    repair was attempted, and the result carries no errors.
    """
    results = _results(results)
    if not results:
        return 0.0
    ok = 0
    for r in results:
        if not getattr(r, "is_valid_geometry", False):
            continue
        if _repair_attempted(r):
            continue
        if getattr(r, "errors", None):
            continue
        ok += 1
    return ok / len(results)


def repair_success_rate(results: list[Any]) -> float:
    """Fraction of repaired results that ended up geometry-valid.

    Results that never needed repair are excluded from the denominator;
    when nothing was repaired the rate is 0.0.
    """
    results = _results(results)
    repaired = [r for r in results if _repair_attempted(r)]
    if not repaired:
        return 0.0
    return sum(1.0 for r in repaired if getattr(r, "is_valid_geometry", False)) / len(repaired)


def _attempts_of(result: Any) -> int:
    """Iterations-to-outcome for one result (1 for first-try success)."""
    iterations = getattr(result, "iterations", None)
    if isinstance(iterations, int) and iterations > 0:
        return iterations
    attempts = getattr(result, "attempts", None)
    if isinstance(attempts, int) and attempts > 0:
        return attempts
    report = getattr(result, "repair_report", None)
    if isinstance(report, dict):
        iterations = report.get("iterations") or report.get("attempts")
        if isinstance(iterations, int) and iterations > 0:
            return iterations
    return 1


def iterations_to_success(results: list[Any]) -> float:
    """Mean iterations across results that reached a valid state.

    A result counts toward the mean when it is geometry-valid; results
    that never became valid are excluded (they belong to the failure rate).
    """
    results = _results(results)
    succeeded = [r for r in results if getattr(r, "is_valid_geometry", False)]
    if not succeeded:
        return 0.0
    return sum(_attempts_of(r) for r in succeeded) / len(succeeded)


def mean_attempts(results: list[Any]) -> float:
    """Mean attempt count across ALL results (valid or not)."""
    results = _results(results)
    if not results:
        return 0.0
    return sum(_attempts_of(r) for r in results) / len(results)


def run_execution_benchmark(
    results: list[Any],
    actual_costs: list[float] | None = None,
    safety_required: float = 1.5,
) -> dict[str, Any]:
    """Aggregate the standard execution metrics into one report."""
    estimated = [float(getattr(r, "estimated_cost_usd", 0.0)) for r in _results(results)]
    report: dict[str, Any] = {
        "geometry_validity_rate": round(geometry_validity_rate(results), 4),
        "manufacturability_rate": round(manufacturability_rate(results), 4),
        "safety_factor_pass_rate": round(safety_factor_pass_rate(results, safety_required), 4),
        "simulation_pass_rate": round(simulation_pass_rate(results), 4),
        "confidence_agreement": round(confidence_agreement(results), 4),
        "repair_rate": round(repair_rate(results), 4),
        "initial_success_rate": round(initial_success_rate(results), 4),
        "repair_success_rate": round(repair_success_rate(results), 4),
        "iterations_to_success": round(iterations_to_success(results), 4),
        "mean_attempts": round(mean_attempts(results), 4),
        "checks": {
            "results": len(_results(results)),
            "actual_costs": len(actual_costs or []),
        },
    }
    if actual_costs is not None and len(actual_costs) == len(estimated):
        report["cost_error"] = round(cost_error(estimated, actual_costs), 4)
    else:
        report["cost_error"] = 0.0
    return report


__all__ = [
    "confidence_agreement",
    "cost_error",
    "geometry_validity_rate",
    "initial_success_rate",
    "iterations_to_success",
    "manufacturability_rate",
    "mean_attempts",
    "repair_rate",
    "repair_success_rate",
    "run_execution_benchmark",
    "safety_factor_pass_rate",
    "simulation_pass_rate",
]
