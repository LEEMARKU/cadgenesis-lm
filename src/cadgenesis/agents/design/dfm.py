"""cadgenesis.agents.design.dfm
=============================
DFM manufacturing agent for the design swarm.

:class:`DFMManufacturingAgent` runs the Design-for-Manufacturing rule engine
(:class:`cadgenesis.reasoning.manufacturing_rules.ManufacturingRules`) and —
unlike the legacy :class:`~cadgenesis.agents.manufacturing.ManufacturingAgent`
— also ranks every known process for a part, letting the orchestration loop
switch processes autonomously when the current one is not manufacturable.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.agents.base import AgentRequest, AgentResult
from cadgenesis.agents.infrastructure import AgentBase, Capability
from cadgenesis.reasoning.manufacturing_rules import ManufacturingRules

_PROCESS_CHECK = {
    "machining": "check_machining",
    "injection_molding": "check_injection_molding",
    "3d_printing": "check_3d_printing",
    "sheet_metal": "check_sheet_metal",
    "casting": "check_casting",
    "welding": "check_welding",
    "tooling": "check_tooling",
}


class DFMManufacturingAgent(AgentBase):
    """Assesses manufacturability and recommends the best process."""

    role = "dfm_manufacturing"
    actions = ("assess", "recommend_process")
    version = "1.0.0"
    capabilities = (
        Capability(
            "dfm.assess",
            "run DFM rule checks for a part and its processes",
            inputs=("part", "processes"),
            outputs=("passed", "checks", "errors", "warnings"),
        ),
        Capability(
            "dfm.recommend_process",
            "rank viable manufacturing processes for a part",
            inputs=("part",),
            outputs=("scores", "recommended"),
        ),
    )

    def __init__(self, rules: ManufacturingRules | None = None) -> None:
        super().__init__()
        self.rules = rules or ManufacturingRules()

    def process(self, request: AgentRequest) -> AgentResult:
        try:
            if request.action == "assess":
                return self._assess(request)
            if request.action == "recommend_process":
                return self._recommend_process(request)
            return self._fail(request, f"unsupported action {request.action!r}")
        except (KeyError, TypeError, ValueError) as exc:
            return self._fail(request, f"{type(exc).__name__}: {exc}")

    # ----------------------------------------------------------------- assess

    def _assess(self, request: AgentRequest) -> AgentResult:
        part = request.payload.get("part")
        if not isinstance(part, dict):
            return self._fail(request, "assess requires a 'part' dictionary")
        part_data = dict(part)
        processes = request.payload.get("processes")
        if processes is not None:
            part_data["processes"] = list(processes)
        assessment = self.rules.assess(part_data)
        output = {
            "passed": assessment.passed,
            "checks": [
                {
                    "check": check.check,
                    "passed": check.is_passed,
                    "severity": check.severity,
                    "detail": check.detail,
                    "recommendation": check.recommendation,
                }
                for check in assessment.checks
            ],
            "errors": [check.check for check in assessment.errors],
            "warnings": [check.check for check in assessment.warnings],
            "recommendations": [
                check.recommendation
                for check in assessment.checks
                if not check.is_passed and check.recommendation
            ],
            "summary": assessment.summary(),
        }
        return AgentResult(
            self.role,
            request.action,
            ok=assessment.passed,
            output=output,
            message=(
                "DFM assessment passed"
                if assessment.passed
                else (f"{len(assessment.errors)} DFM errors, {len(assessment.warnings)} warnings")
            ),
            task_id=request.task_id,
        )

    # ------------------------------------------------------ process ranking

    def _recommend_process(self, request: AgentRequest) -> AgentResult:
        part = request.payload.get("part")
        if not isinstance(part, dict):
            return self._fail(request, "recommend_process requires a 'part' dictionary")
        scored: list[dict[str, Any]] = []
        for process, check_name in _PROCESS_CHECK.items():
            handler = getattr(self.rules, check_name, None)
            if handler is None:
                continue
            checks = handler(part)
            errors = [c for c in checks if not c.is_passed and c.severity == "error"]
            warnings = [c for c in checks if not c.is_passed and c.severity != "error"]
            scored.append(
                {
                    "process": process,
                    "errors": len(errors),
                    "warnings": len(warnings),
                    "viable": not errors,
                    "failed_checks": [c.check for c in errors],
                }
            )
        scored.sort(key=lambda row: (row["errors"], row["warnings"]))
        recommended = next((row["process"] for row in scored if row["viable"]), None)
        return AgentResult(
            self.role,
            request.action,
            ok=True,
            output={"scores": scored, "recommended": recommended},
            message=(
                f"recommended process: {recommended}" if recommended else "no viable process found"
            ),
            task_id=request.task_id,
        )

    # ----------------------------------------------------------------- misc

    def _fail(self, request: AgentRequest, message: str) -> AgentResult:
        return AgentResult(self.role, request.action, False, {}, message, request.task_id)


__all__ = ["DFMManufacturingAgent"]
