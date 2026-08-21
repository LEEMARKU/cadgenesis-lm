"""cadgenesis.agents.manufacturing
================================
Specialised manufacturing agent: Design-for-Manufacturing (DFM) assessment.
"""

from __future__ import annotations

from cadgenesis.agents.base import Agent, AgentRequest, AgentResult
from cadgenesis.reasoning.manufacturing_rules import ManufacturingRules


class ManufacturingAgent(Agent):
    """Checks a part against machining / molding / printing / sheet-metal rules."""

    role = "manufacturing"
    actions = ("assess", "check_process")

    def __init__(self, rules: ManufacturingRules | None = None) -> None:
        super().__init__()
        self.rules = rules or ManufacturingRules()

    def process(self, request: AgentRequest) -> AgentResult:
        payload = request.payload
        part = payload.get("part")
        if not isinstance(part, dict):
            return AgentResult(
                role=self.role,
                action=request.action,
                ok=False,
                message="assess requires a 'part' dictionary",
                task_id=request.task_id,
            )
        if request.action == "check_process":
            process = payload.get("process")
            if process not in ("machining", "injection_molding", "3d_printing", "sheet_metal"):
                return AgentResult(
                    role=self.role,
                    action=request.action,
                    ok=False,
                    message=f"unknown process {process!r}",
                    task_id=request.task_id,
                )
            part = dict(part)
            part["processes"] = [process]
        try:
            assessment = self.rules.assess(part)
            summary = assessment.summary()
            return AgentResult(
                role=self.role,
                action=request.action,
                ok=assessment.passed,
                output={
                    "passed": assessment.passed,
                    "checks": [
                        {
                            "check": check.check,
                            "passed": check.is_passed,
                            "detail": check.detail,
                        }
                        for check in assessment.checks
                    ],
                    "summary": summary,
                },
                message=(
                    "manufacturability OK"
                    if assessment.passed
                    else f"{len(assessment.errors)} DFM errors"
                ),
                task_id=request.task_id,
            )
        except (KeyError, ValueError, TypeError) as exc:
            return AgentResult(
                role=self.role,
                action=request.action,
                ok=False,
                message=str(exc),
                task_id=request.task_id,
            )
