"""cadgenesis.evaluation.agent_metrics
====================================
Multi-agent metrics (Pillar 5).

Pure-function metrics for the agent platform: task success, error-rate,
latency, consensus agreement, fleet coverage (capability coverage of the
registry), and end-to-end pipeline success.  Every metric takes plain data so
benchmarks can plug in any generator.
"""

from __future__ import annotations

from typing import Any


def task_success_rate(results: list[dict[str, Any]]) -> float:
    """Fraction of agent results with ``ok == True``."""
    if not results:
        return 0.0
    return sum(1.0 for r in results if r.get("ok")) / len(results)


def error_rate(results: list[dict[str, Any]]) -> float:
    """Fraction of agent results with ``ok == False``."""
    return 1.0 - task_success_rate(results)


def mean_latency(durations: list[float]) -> float:
    """Mean per-task duration in seconds (0.0 for empty input)."""
    if not durations:
        return 0.0
    return sum(durations) / len(durations)


def p95_latency(durations: list[float]) -> float:
    """95th-percentile latency; ``max`` for few samples."""
    if not durations:
        return 0.0
    ordered = sorted(durations)
    index = max(0, int(0.95 * (len(ordered) - 1)))
    return ordered[index]


def consensus_agreement(summaries: list[dict[str, Any]]) -> float:
    """Fraction of consensus summaries whose decision matches the majority."""
    if not summaries:
        return 0.0
    matched = sum(1 for s in summaries if s.get("decision") == s.get("majority"))
    return matched / len(summaries)


def fleet_coverage(registry_snapshot: list[dict[str, Any]]) -> dict[str, Any]:
    """Capability coverage of the registry.

    ``registry_snapshot`` is a list of ``{"role", "capabilities"}`` dicts.
    """
    roles = [a["role"] for a in registry_snapshot]
    capability_count = sum(len(a.get("capabilities", [])) for a in registry_snapshot)
    return {
        "role_count": len(roles),
        "capability_count": capability_count,
        "avg_capabilities_per_role": (capability_count / len(roles) if roles else 0.0),
    }


def pipeline_success(reports: list[dict[str, Any]]) -> float:
    """Fraction of pipeline reports whose validation gate passed."""
    if not reports:
        return 0.0
    return sum(1.0 for r in reports if r.get("validation", {}).get("passed")) / len(reports)


def run_agent_benchmark(
    results: list[dict[str, Any]],
    durations: list[float],
    summaries: list[dict[str, Any]],
    registry_snapshot: list[dict[str, Any]],
    pipeline_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate the standard agent metrics into one report."""
    return {
        "task_success_rate": task_success_rate(results),
        "error_rate": error_rate(results),
        "mean_latency_s": mean_latency(durations),
        "p95_latency_s": p95_latency(durations),
        "consensus_agreement": consensus_agreement(summaries),
        "pipeline_success": pipeline_success(pipeline_reports),
        "coverage": fleet_coverage(registry_snapshot),
        "checks": {
            "results": len(results),
            "consensus": len(summaries),
            "pipelines": len(pipeline_reports),
        },
    }


__all__ = [
    "consensus_agreement",
    "error_rate",
    "fleet_coverage",
    "mean_latency",
    "p95_latency",
    "pipeline_success",
    "run_agent_benchmark",
    "task_success_rate",
]
