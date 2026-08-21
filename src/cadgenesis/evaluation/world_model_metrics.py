"""cadgenesis.evaluation.world_model_metrics
============================================
World-model metrics (Pillar 4).

Metrics for the world-model reasoning stack: spatial check accuracy,
mechanical-safety conformance, assembly integrity, affordance coverage,
path-collision detection and end-to-end planning success.  Every metric is a
pure function that takes ``(predictions, ground_truth)`` pairs so benchmarks
can plug in any generator.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.world_model.objects import WorldObject


def accuracy(predictions: list[bool], ground_truth: list[bool]) -> float:
    """Fraction of predictions that match the ground truth."""
    if not predictions:
        return 0.0
    pairs = zip(predictions, ground_truth, strict=True)
    return sum(1.0 for p, g in pairs if bool(p) == bool(g)) / len(predictions)


def safety_margin(
    predicted_fos: list[float],
    required_fos: list[float],
) -> float:
    """Mean factor-of-safety headroom: ``mean(predicted - required)``."""
    if not predicted_fos:
        return 0.0
    return sum(p - r for p, r in zip(predicted_fos, required_fos, strict=True)) / len(predicted_fos)


def assembly_integrity(
    assemblies: list[list[dict[str, Any]]],
) -> float:
    """Fraction of assembly check-lists that pass entirely."""
    if not assemblies:
        return 0.0
    return sum(1.0 for checks in assemblies if all(c.get("passed") for c in checks)) / len(
        assemblies
    )


def affordance_coverage_with(
    mapper: Any,
    objects: list[WorldObject],
    required_actions: list[list[str]],
) -> float:
    """Fraction of required actions covered, given a concrete mapper."""
    if not objects:
        return 0.0
    covered = 0
    total = 0
    for obj, required in zip(objects, required_actions, strict=True):
        actual = {a.action for a in mapper.affordances(obj)}
        covered += len(actual & set(required))
        total += len(required)
    return covered / total if total else 1.0


def path_collision_detection(
    predictions: list[bool],
    ground_truth: list[bool],
) -> float:
    """Accuracy of collision-free flags (True = predicted collision-free)."""
    return accuracy(predictions, ground_truth)


def planning_success(
    outcomes: list[dict[str, Any]],
) -> float:
    """Fraction of plan executions where every step passed."""
    if not outcomes:
        return 0.0
    return sum(1.0 for o in outcomes if o.get("all_passed")) / len(outcomes)


def run_world_benchmark(
    spatial_checks: list[tuple[bool, bool]],
    safety_checks: list[tuple[float, float]],
    assembly_checks: list[list[dict[str, Any]]],
    path_checks: list[tuple[bool, bool]],
    plan_outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate the standard world-model metrics into one report."""
    return {
        "spatial_accuracy": (
            accuracy([p for p, _ in spatial_checks], [g for _, g in spatial_checks])
            if spatial_checks
            else 0.0
        ),
        "safety_margin": (
            safety_margin([p for p, _ in safety_checks], [r for _, r in safety_checks])
            if safety_checks
            else 0.0
        ),
        "assembly_integrity": assembly_integrity(assembly_checks),
        "path_collision_accuracy": (
            path_collision_detection([p for p, _ in path_checks], [g for _, g in path_checks])
            if path_checks
            else 0.0
        ),
        "planning_success": planning_success(plan_outcomes),
        "checks": {
            "spatial": len(spatial_checks),
            "safety": len(safety_checks),
            "assembly": len(assembly_checks),
            "path": len(path_checks),
            "plans": len(plan_outcomes),
        },
    }


__all__ = [
    "accuracy",
    "affordance_coverage_with",
    "assembly_integrity",
    "path_collision_detection",
    "planning_success",
    "run_world_benchmark",
    "safety_margin",
]
