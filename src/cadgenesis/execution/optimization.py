"""cadgenesis.execution.optimization
==================================
Design optimization engine for the CAD execution pipeline.

Scores design alternatives across engineering objectives — weight, material
suitability, complexity, additive print time, cost and structural efficiency —
and emits concrete improvement suggestions for a given design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

OBJECTIVES = ("weight", "material", "complexity", "print_time", "cost", "structural")


@dataclass
class OptimizationReport:
    """Result of a design-optimization pass."""

    passed: bool
    scores: list[dict[str, Any]] = field(default_factory=list)
    best_index: int = -1
    suggestions: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        best = self.scores[self.best_index] if 0 <= self.best_index < len(self.scores) else None
        return {
            "passed": self.passed,
            "best_index": self.best_index,
            "best_score": round(best["score"], 4) if best else None,
            "candidates": len(self.scores),
            "suggestions": list(self.suggestions),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.summary()


class OptimizationEngine:
    """Multi-objective design scoring and improvement suggestions."""

    DEFAULT_WEIGHTS = {
        "weight": 0.25,
        "material": 0.15,
        "complexity": 0.1,
        "print_time": 0.1,
        "cost": 0.25,
        "structural": 0.15,
    }

    def __init__(self) -> None:
        self.weights = dict(self.DEFAULT_WEIGHTS)

    # --------------------------------------------------------------- scoring

    def score_designs(
        self,
        designs: list[dict[str, Any]],
        weights: dict[str, float] | None = None,
    ) -> OptimizationReport:
        """Score candidate designs (higher = better, normalized 0..1).

        Each design carries objective metrics: ``weight_kg``, ``cost_usd``,
        ``print_time_h``, ``feature_count`` (complexity), ``material_rank``
        (1 = best) and ``structural_efficiency`` (0..1).
        """
        w = dict(self.DEFAULT_WEIGHTS)
        if weights:
            w.update(weights)
        report = OptimizationReport(passed=True)
        if not designs:
            report.passed = False
            report.messages.append("no design candidates")
            return report

        raw = {objective: [_metric(d, objective) for d in designs] for objective in OBJECTIVES}
        maxima = {objective: max(values) for objective, values in raw.items()}
        for index, design in enumerate(designs):
            score = 0.0
            detail: dict[str, float] = {}
            for objective, values in raw.items():
                value = values[index]
                span = maxima[objective] or 1.0
                normalized = value / span
                # lower-is-better objectives flip the normalized value
                if objective in ("weight", "material", "complexity", "print_time", "cost"):
                    normalized = 1.0 - normalized
                weight = w.get(objective, 0.0)
                score += weight * normalized
                detail[objective] = round(normalized, 4)
            report.scores.append(
                {
                    "index": index,
                    "name": str(design.get("name") or f"candidate-{index}"),
                    "score": round(score, 4),
                    "objectives": detail,
                }
            )
        report.scores.sort(key=lambda s: (-s["score"], s["index"]))
        report.best_index = report.scores[0]["index"]
        return report

    # ------------------------------------------------------------ suggestions

    def suggest(self, design: dict[str, Any]) -> list[str]:
        """Heuristic improvement suggestions for one design."""
        suggestions: list[str] = []
        wall = design.get("wall_thickness_mm")
        if wall is not None and float(wall) > 5.0:
            suggestions.append(f"reduce wall thickness from {wall} mm to ~2-3 mm to save weight")
        ribs = design.get("ribs")
        if ribs:
            suggestions.append("add ribbing to increase stiffness without adding wall mass")
        hollow = design.get("hollow")
        if not hollow and design.get("feature_count", 1) > 1:
            suggestions.append("consider a shell/hollow core to cut material volume")
        material = design.get("material", {})
        name = str(material.get("name") or "").lower()
        if name == "steel" and design.get("weight_kg"):
            suggestions.append("review if an aluminum alloy can meet the load case (≈60% lighter)")
        if design.get("print_time_h"):
            suggestions.append("increase layer height or reduce infill to lower print time")
        if not suggestions:
            suggestions.append("design is already lean; no immediate structural savings found")
        return suggestions

    def optimize(
        self,
        design: dict[str, Any],
        objectives: list[str] | None = None,
        weights: dict[str, float] | None = None,
    ) -> OptimizationReport:
        """Run suggestions + score a single design against objectives."""
        report = OptimizationReport(passed=True)
        report.suggestions = self.suggest(design)
        wanted = [o for o in (objectives or OBJECTIVES) if o in OBJECTIVES]
        if not wanted:
            report.passed = False
            report.messages.append("no recognized objectives")
            return report
        variants = [design]
        for objective in wanted:
            variant = dict(design)
            variant["name"] = f"{design.get('name', 'design')}-{objective}"
            variants.append(variant)
        scored = self.score_designs(variants, weights=weights)
        report.scores = scored.scores
        report.best_index = scored.best_index
        report.messages = list(scored.messages)
        return report

    def summary(self) -> dict[str, Any]:
        return {"objectives": list(OBJECTIVES), "weights": self.weights}


def _metric(design: dict[str, Any], objective: str) -> float:
    """Extract a raw objective metric from a design dict (0.0 when absent)."""
    keys = {
        "weight": ("weight_kg",),
        "material": ("material_rank",),
        "complexity": ("feature_count",),
        "print_time": ("print_time_h",),
        "cost": ("cost_usd",),
        "structural": ("structural_efficiency",),
    }[objective]
    for key in keys:
        value = design.get(key)
        if value is not None:
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


__all__ = [
    "OBJECTIVES",
    "OptimizationEngine",
    "OptimizationReport",
]
