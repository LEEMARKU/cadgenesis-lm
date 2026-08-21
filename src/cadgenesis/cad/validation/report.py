"""cadgenesis.cad.validation.report
================================
Validation report types for the CAD validation pipeline.

``CadCheckResult`` mirrors the reasoning toolkit's ``CheckResult`` while
``CadValidationReport`` aggregates results and exposes a summary suitable
for downstream tooling (benchmarks, agents, CLI output).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CadCheckResult:
    """Outcome of a single CAD validation check."""

    name: str
    passed: bool
    severity: str = "error"
    detail: str = ""
    recommendation: str = ""

    @property
    def is_passed(self) -> bool:
        return self.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "detail": self.detail,
            "recommendation": self.recommendation,
        }


@dataclass
class CadValidationReport:
    """Aggregated result of a validation pipeline run."""

    results: list[CadCheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def errors(self) -> list[CadCheckResult]:
        return [r for r in self.results if not r.passed and r.severity == "error"]

    @property
    def warnings(self) -> list[CadCheckResult]:
        return [r for r in self.results if not r.passed and r.severity != "error"]

    def failed(self) -> list[CadCheckResult]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "total": len(self.results),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "failed": [r.name for r in self.results if not r.passed],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "results": [r.to_dict() for r in self.results],
        }


__all__ = ["CadCheckResult", "CadValidationReport"]
