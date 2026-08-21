"""cadgenesis.agents.validation
=============================
Specialised validation agent: orchestrates design validation checks.
"""

from __future__ import annotations

from cadgenesis.agents.base import Agent, AgentRequest, AgentResult
from cadgenesis.reasoning.validator import DesignValidator


class ValidationAgent(Agent):
    """Runs rule / constraint / geometry / manufacturing / topology checks."""

    role = "validation"
    actions = ("validate", "report")

    def __init__(self, validator: DesignValidator | None = None) -> None:
        super().__init__()
        self.validator = validator or DesignValidator()

    def process(self, request: AgentRequest) -> AgentResult:
        payload = request.payload
        context = payload.get("context")
        if not isinstance(context, dict):
            return AgentResult(
                role=self.role,
                action=request.action,
                ok=False,
                message="validate requires a 'context' dictionary",
                task_id=request.task_id,
            )
        try:
            report = self.validator.validate(context)
            if request.action == "report":
                return AgentResult(
                    role=self.role,
                    action=request.action,
                    ok=report.passed,
                    output=report.summary(),
                    message=(
                        "validation passed"
                        if report.passed
                        else f"{len(report.errors)} errors, {len(report.warnings)} warnings"
                    ),
                    task_id=request.task_id,
                )
            return AgentResult(
                role=self.role,
                action=request.action,
                ok=report.passed,
                output={"passed": report.passed, "summary": report.summary()},
                message=("validation passed" if report.passed else "validation failed"),
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
