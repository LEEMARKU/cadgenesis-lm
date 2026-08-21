"""cadgenesis.reasoning.validator
================================
Design validator: orchestrates rule, constraint, geometry, manufacturing,
topology and symbolic checks into a single :class:`ValidationReport`.

This is the entry point application code should use to ask "is this design
sound?" before committing it to the execution pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_VALID_CATEGORIES = (
    "rule",
    "constraint",
    "geometry",
    "manufacturing",
    "topology",
    "symbolic",
    "custom",
)


@dataclass
class CheckResult:
    """A single validation check outcome."""

    category: str
    name: str
    passed: bool
    severity: str = "error"
    detail: str = ""
    recommendation: str = ""

    def __post_init__(self) -> None:
        if self.category not in _VALID_CATEGORIES:
            raise ValueError(f"invalid category {self.category!r}")


@dataclass
class ValidationReport:
    """Aggregated outcome of a validation run."""

    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def errors(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == "error"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity != "error"]

    def by_category(self) -> dict[str, list[CheckResult]]:
        grouped: dict[str, list[CheckResult]] = {}
        for result in self.results:
            grouped.setdefault(result.category, []).append(result)
        return grouped

    def summary(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "total": len(self.results),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "by_category": {
                category: sum(1 for r in self.results if r.category == category)
                for category in _VALID_CATEGORIES
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "results": [
                {
                    "category": r.category,
                    "name": r.name,
                    "passed": r.passed,
                    "severity": r.severity,
                    "detail": r.detail,
                    "recommendation": r.recommendation,
                }
                for r in self.results
            ],
        }


class DesignValidator:
    """Runs a configurable set of checks against a design context."""

    def __init__(
        self,
        rule_engine=None,
        manufacturing_rules=None,
        geometry_reasoner=None,
        constraint_solver=None,
        topology_analyzer=None,
    ) -> None:
        self.rule_engine = rule_engine
        self.manufacturing_rules = manufacturing_rules
        self.geometry_reasoner = geometry_reasoner
        self.constraint_solver = constraint_solver
        self.topology_analyzer = topology_analyzer
        self._custom_checks: list[Callable[[dict[str, Any]], list[CheckResult]]] = []

    def add_check(self, check_fn: Callable[[dict[str, Any]], list[CheckResult]]) -> None:
        """Register an extra check returning a list of ``CheckResult``."""
        self._custom_checks.append(check_fn)

    # ------------------------------------------------------------ built-ins

    def _rule_checks(self, context: dict[str, Any]) -> list[CheckResult]:
        if self.rule_engine is None:
            return []
        results: list[CheckResult] = []
        for result in self.rule_engine.evaluate(context):
            if not result.triggered:
                continue
            results.append(
                CheckResult(
                    category="rule",
                    name=result.name,
                    passed=result.rule.severity_index() < 2,  # info/warning pass
                    severity=result.rule.severity,
                    detail=result.message or result.rule.description,
                    recommendation=result.rule.meta.get("recommendation", ""),
                )
            )
        return results

    def _constraint_checks(self, context: dict[str, Any]) -> list[CheckResult]:
        if self.constraint_solver is None:
            return []
        variables = context.get("constraint_variables")
        constraints = context.get("constraints")
        if not variables or not constraints:
            return []
        solution = self.constraint_solver.solve(variables, constraints)
        results = [
            CheckResult(
                category="constraint",
                name="constraint_system_feasible",
                passed=solution.feasible,
                detail=f"max residual {solution.max_residual:.2e}",
                recommendation="; ".join(solution.messages) or "Adjust dimensional bounds.",
            )
        ]
        for name, residual in solution.residuals.items():
            if abs(residual) > 1e-6:
                results.append(
                    CheckResult(
                        category="constraint",
                        name=f"constraint:{name}",
                        passed=False,
                        detail=f"residual {residual:.2e}",
                    )
                )
        return results

    def _geometry_checks(self, context: dict[str, Any]) -> list[CheckResult]:
        reasoner = self.geometry_reasoner
        if reasoner is None:
            return []
        results: list[CheckResult] = []
        for primitive in context.get("primitives", []):
            check = reasoner.validate(primitive)
            results.append(
                CheckResult(
                    category="geometry",
                    name=f"geometry:{primitive.name or primitive.kind}",
                    passed=check.valid,
                    detail="; ".join(check.messages) or "dimensions OK",
                )
            )
        if "interference_pairs" in context:
            results.extend(
                CheckResult(
                    category="geometry",
                    name=f"interference:{pair[0].name}-{pair[1].name}",
                    passed=not reasoner.overlaps(pair[0], pair[1]),
                    severity="warning",
                    detail="AABBs overlap",
                    recommendation="Move the parts or reduce their size.",
                )
                for pair in context["interference_pairs"]
            )
        return results

    def _manufacturing_checks(self, context: dict[str, Any]) -> list[CheckResult]:
        rules = self.manufacturing_rules
        if rules is None:
            return []
        part = context.get("part")
        if part is None:
            return []
        assessment = rules.assess(part)
        return [
            CheckResult(
                category="manufacturing",
                name=f"mfg:{check.check}",
                passed=check.passed,
                severity=check.severity,
                detail=check.detail,
                recommendation=check.recommendation,
            )
            for check in assessment.checks
        ]

    def _topology_checks(self, context: dict[str, Any]) -> list[CheckResult]:
        analyzer = self.topology_analyzer
        if analyzer is None:
            return []
        stats_data = context.get("topology")
        if stats_data is None:
            return []
        stats = analyzer.analyze(**stats_data)
        results = [
            CheckResult(
                category="topology",
                name="euler_poincare",
                passed=not stats.notes,
                detail="; ".join(stats.notes) or "Euler-Poincare satisfied",
            )
        ]
        if stats.genus < 0:
            results.append(
                CheckResult(
                    category="topology",
                    name="genus_non_negative",
                    passed=False,
                    detail=f"negative genus {stats.genus}",
                )
            )
        return results

    # ------------------------------------------------------------------ run

    def validate(self, context: dict[str, Any]) -> ValidationReport:
        """Run all registered checks against ``context``.

        ``context`` keys consumed by the built-ins:

        - ``primitives`` (list of :class:`Primitive`)
        - ``interference_pairs`` (list of 2-tuples of primitives)
        - ``constraint_variables`` / ``constraints``
        - ``part`` (dict for DFM checks)
        - ``topology`` (dict of counts for ``TopologyAnalyzer.analyze``)
        """
        report = ValidationReport()
        report.results.extend(self._rule_checks(context))
        report.results.extend(self._constraint_checks(context))
        report.results.extend(self._geometry_checks(context))
        report.results.extend(self._manufacturing_checks(context))
        report.results.extend(self._topology_checks(context))
        for check_fn in self._custom_checks:
            report.results.extend(check_fn(context))
        return report

    def validate_plan(self, plan) -> ValidationReport:
        """Validate a :class:`cadgenesis.reasoning.planner.CADPlan`."""
        problems = plan.validate()
        return ValidationReport(
            results=[
                CheckResult(
                    category="custom",
                    name="plan_soundness",
                    passed=not problems,
                    detail="; ".join(problems) or "plan is sound",
                )
            ]
        )


__all__ = [
    "CheckResult",
    "DesignValidator",
    "ValidationReport",
]
