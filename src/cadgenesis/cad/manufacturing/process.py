"""cadgenesis.cad.manufacturing.process
=====================================
Manufacturing process selection engine.

Given a part (material category + geometric / DFM descriptors) the engine
suggests the most appropriate primary processes and flags DFM feasibility.
This complements the existing :mod:`cadgenesis.reasoning.manufacturing_rules`
DFM checker with a *native* part→process mapping layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# process -> (material categories, part-size hints)
_CAPABILITY: dict[str, dict[str, Any]] = {
    "cnc_milling": {"materials": {"metal", "plastic", "composite"}, "min_part_size_mm": 5.0},
    "cnc_turning": {"materials": {"metal", "plastic", "composite"}, "min_part_size_mm": 3.0},
    "3d_printing_fdm": {"materials": {"plastic"}, "min_part_size_mm": 2.0},
    "3d_printing_sla": {"materials": {"plastic", "ceramic"}, "min_part_size_mm": 0.2},
    "3d_printing_sls": {"materials": {"plastic"}, "min_part_size_mm": 0.5},
    "3d_printing_dmls": {"materials": {"metal"}, "min_part_size_mm": 0.5},
    "casting_sand": {"materials": {"metal"}, "min_part_size_mm": 25.0},
    "casting_die": {"materials": {"metal"}, "min_part_size_mm": 10.0},
    "casting_investment": {"materials": {"metal"}, "min_part_size_mm": 3.0},
    "injection_molding": {"materials": {"plastic"}, "min_part_size_mm": 5.0},
    "sheet_metal": {"materials": {"metal"}, "min_part_size_mm": 10.0},
    "welding": {"materials": {"metal"}, "min_part_size_mm": 20.0},
}

_PROCESS_TO_GROUP = {
    "cnc_milling": "cnc",
    "cnc_turning": "cnc",
    "3d_printing_fdm": "3d_printing",
    "3d_printing_sla": "3d_printing",
    "3d_printing_sls": "3d_printing",
    "3d_printing_dmls": "3d_printing",
    "casting_sand": "casting",
    "casting_die": "casting",
    "casting_investment": "casting",
    "injection_molding": "injection_molding",
    "sheet_metal": "sheet_metal",
    "welding": "welding",
}


@dataclass
class ProcessSuggestion:
    """A recommended manufacturing process with a score."""

    process: str
    group: str
    score: float  # 0..1 (1 = ideal fit)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "process": self.process,
            "group": self.group,
            "score": self.score,
            "reasons": self.reasons,
        }


@dataclass
class ProcessSelection:
    """Result of the process-selection analysis."""

    suggestions: list[ProcessSuggestion]

    @property
    def best(self) -> ProcessSuggestion | None:
        return self.suggestions[0] if self.suggestions else None

    def by_group(self) -> dict[str, list[ProcessSuggestion]]:
        grouped: dict[str, list[ProcessSuggestion]] = {}
        for suggestion in self.suggestions:
            grouped.setdefault(suggestion.group, []).append(suggestion)
        return grouped


class ProcessSelector:
    """Selects manufacturing processes from part descriptors.

    ``part`` may carry: ``material_category``, ``batch_size``,
    ``max_part_size_mm``, ``min_wall_thickness``, ``draft_angle``,
    ``has_undercut``, ``tight_tolerance`` and ``required_group``.
    """

    def __init__(self) -> None:
        self.capabilities = {name: dict(info) for name, info in _CAPABILITY.items()}

    def select(self, part: dict[str, Any]) -> ProcessSelection:
        category = str(part.get("material_category", "plastic"))
        batch = float(part.get("batch_size", 100))
        max_size = float(part.get("max_part_size_mm", 100.0))
        required_group = part.get("required_group")

        suggestions: list[ProcessSuggestion] = []
        for process, info in self.capabilities.items():
            if required_group and _PROCESS_TO_GROUP[process] != required_group:
                continue
            score = 1.0
            reasons: list[str] = []
            if category not in info["materials"]:
                score -= 1.0
                reasons.append(f"unsuitable for {category}")
            if max_size < info["min_part_size_mm"]:
                score -= 0.4
                reasons.append("part smaller than process minimum")
            if process == "injection_molding":
                if batch < 1000:
                    score -= 0.4
                    reasons.append("small batch: mould cost not justified")
                if part.get("draft_angle") is not None and float(part["draft_angle"]) < 1.0:
                    score -= 0.3
                    reasons.append("insufficient draft angle")
                if part.get("has_undercut"):
                    score -= 0.3
                    reasons.append("undercut requires side-action core")
            if process == "3d_printing_fdm" and batch > 100:
                score -= 0.5
                reasons.append("additive is slow for large batches")
            if (
                process in ("3d_printing_dmls", "3d_printing_sls", "casting_investment")
                and batch > 1000
            ):
                score -= 0.3
                reasons.append("unit cost high for very large batch")
            if process == "cnc_milling" and part.get("tight_tolerance"):
                score += 0.2
                reasons.append("machining holds tight tolerances")
            score = max(0.0, min(1.0, score))
            if score > 0.0:
                suggestions.append(
                    ProcessSuggestion(process, _PROCESS_TO_GROUP[process], round(score, 3), reasons)
                )
        suggestions.sort(key=lambda s: (-s.score, s.process))
        return ProcessSelection(suggestions)


__all__ = ["ProcessSelection", "ProcessSelector", "ProcessSuggestion"]
