"""cadgenesis.execution.manufacturing
===================================
Manufacturability analysis for the CAD execution pipeline.

Composes the existing manufacturing substrate — the DFM rule engine
(`reasoning.manufacturing_rules.ManufacturingRules`), the process selection
engine (`cad.manufacturing.process.ProcessSelector`) and the manufacturing
feature vocabulary (`cad.manufacturing.features`) — into one execution-layer
analyser covering CNC, additive, casting, injection moulding, sheet metal,
welding and tooling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from cadgenesis.cad.manufacturing.features import (
    PROCESS_GROUPS,
    ManufacturingFeature,
)
from cadgenesis.cad.manufacturing.process import (
    ProcessSelection,
    ProcessSelector,
)
from cadgenesis.reasoning.manufacturing_rules import ManufacturingRules

_CHECK_PREFIXES = {
    "cnc": "machining",
    "3d_printing": "print",
    "casting": "cast",
    "injection_molding": "mold",
    "sheet_metal": "sheet",
    "welding": "weld",
}


@dataclass
class ManufacturingCheck:
    """Single manufacturability check result."""

    name: str
    passed: bool
    severity: str = "error"
    detail: str = ""
    recommendation: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "detail": self.detail,
            "recommendation": self.recommendation,
            "params": self.params,
        }


@dataclass
class ManufacturingReport:
    """Aggregated manufacturability analysis result."""

    checks: list[ManufacturingCheck] = field(default_factory=list)
    processes: ProcessSelection | None = None

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed(self) -> list[ManufacturingCheck]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> dict[str, Any]:
        best = self.processes.best if self.processes is not None else None
        return {
            "passed": self.passed,
            "total": len(self.checks),
            "failed": [c.name for c in self.checks if not c.passed],
            "best_process": best.process if best else None,
            "process_groups": self.processes.by_group() if self.processes is not None else {},
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "processes": (
                [s.to_dict() for s in self.processes.suggestions]
                if self.processes is not None
                else []
            ),
        }


class ManufacturabilityAnalyzer:
    """Execution-layer manufacturability analyser.

    Wraps the rule engine + process selector; every method returns a
    :class:`ManufacturingReport` and never raises on bad input.
    """

    def __init__(
        self,
        rules: ManufacturingRules | None = None,
        selector: ProcessSelector | None = None,
    ) -> None:
        self.rules = rules or ManufacturingRules()
        self.selector = selector or ProcessSelector()

    def assess(self, part: dict[str, Any]) -> ManufacturingReport:
        """Full DFM assessment: rule checks + process selection."""
        report = ManufacturingReport()
        try:
            assessment = self.rules.assess(part)
        except (TypeError, ValueError, KeyError) as exc:
            report.checks.append(
                ManufacturingCheck("rules:assess", False, detail=f"assessment failed: {exc}")
            )
            return report
        for check in assessment.checks:
            report.checks.append(
                ManufacturingCheck(
                    check.check,
                    bool(check.passed),
                    severity=check.severity,
                    detail=check.detail,
                    recommendation=check.recommendation,
                )
            )
        try:
            report.processes = self.selector.select(part)
        except (TypeError, ValueError, KeyError):
            report.processes = None
        return report

    def assess_group(self, part: dict[str, Any], process_group: str) -> ManufacturingReport:
        """DFM assessment restricted to one process group."""
        if process_group not in PROCESS_GROUPS:
            report = ManufacturingReport()
            report.checks.append(
                ManufacturingCheck(
                    "process:group",
                    False,
                    detail=f"unknown process group {process_group!r}",
                )
            )
            return report
        report = self.assess(part)
        prefix = _CHECK_PREFIXES.get(process_group, process_group)
        report.checks = [c for c in report.checks if c.name.startswith(prefix)]
        report.processes = None
        return report

    def feature_check(
        self,
        features: Sequence[ManufacturingFeature | dict[str, Any]],
    ) -> ManufacturingReport:
        """Validate manufacturing feature descriptors."""
        report = ManufacturingReport()
        for raw in features:
            feature = ManufacturingFeature.from_dict(raw) if isinstance(raw, dict) else raw
            if not isinstance(feature, ManufacturingFeature):
                report.checks.append(
                    ManufacturingCheck(
                        "feature:type",
                        False,
                        detail=f"unsupported feature {type(feature).__name__}",
                    )
                )
                continue
            problems = feature.validate()
            report.checks.append(
                ManufacturingCheck(
                    f"feature:{feature.name}",
                    not problems,
                    detail="; ".join(problems) if problems else "valid",
                )
            )
            report.checks.append(
                ManufacturingCheck(
                    f"feature:{feature.name}:group",
                    feature.process_group in PROCESS_GROUPS,
                    severity="warning",
                    detail=f"group {feature.process_group}",
                )
            )
        return report

    def tolerance_check(self, part: dict[str, Any]) -> ManufacturingReport:
        """Tolerance-feasibility snapshot over the DFM checks."""
        report = self.assess(part)
        report.checks = [
            c for c in report.checks if "tolerance" in c.name or "tolerance" in c.detail.lower()
        ]
        return report

    def summary(self) -> dict[str, Any]:
        return {
            "process_groups": PROCESS_GROUPS,
            "rule_checks": len(self.rules.check_map) if hasattr(self.rules, "check_map") else 0,
        }


__all__ = [
    "ManufacturabilityAnalyzer",
    "ManufacturingCheck",
    "ManufacturingReport",
]
