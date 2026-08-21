"""cadgenesis.agents.safety
==========================
Safety & compliance agent.

Runs the neuro-symbolic :class:`~cadgenesis.reasoning.validator.DesignValidator`
plus a lightweight compliance/hazard gate over a design context and reports
violations with recommendations.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.agents.base import AgentRequest, AgentResult
from cadgenesis.agents.infrastructure import AgentBase, Capability

_HAZARD_RULES = {
    "sharp_edge": "avoid sharp external edges for user-facing parts",
    "entrapment": "no gaps that can entrap fingers (5-25 mm) for consumer parts",
    "burr": "deburr machined features before assembly",
    "pinch_point": "avoid pinch points in linkages and closures",
}


class SafetyComplianceAgent(AgentBase):
    """Checks designs for safety and regulatory compliance."""

    role = "safety"
    actions = ("check", "compliance")
    version = "1.0.0"
    capabilities = (
        Capability("safety.check", "run validator checks over a design context"),
        Capability("safety.compliance", "evaluate a checklist of hazard rules"),
    )

    def __init__(self, validator: Any = None) -> None:
        super().__init__()
        if validator is None:
            from cadgenesis.reasoning.validator import DesignValidator

            validator = DesignValidator()
        self.validator = validator

    def process(self, request: AgentRequest) -> AgentResult:
        payload = request.payload
        if request.action == "check":
            context = payload.get("context")
            if not isinstance(context, dict):
                return self._fail(request, "payload requires 'context' mapping")
            report = self.validator.validate(context)
            return self._ok(
                request,
                {
                    "passed": report.passed,
                    "summary": report.summary(),
                    "errors": [
                        c.to_dict() if hasattr(c, "to_dict") else str(c) for c in report.errors()
                    ],
                    "warnings": [
                        c.to_dict() if hasattr(c, "to_dict") else str(c) for c in report.warnings()
                    ],
                },
                "design validation completed",
            )
        if request.action == "compliance":
            context = payload.get("context", {})
            enabled = payload.get("rules", list(_HAZARD_RULES))
            results = []
            passed_all = True
            for rule in enabled:
                triggered = bool(context.get(f"hazard:{rule}"))
                results.append(
                    {
                        "rule": rule,
                        "guideline": _HAZARD_RULES.get(rule, rule),
                        "triggered": triggered,
                        "passed": not triggered,
                    }
                )
                if triggered:
                    passed_all = False
            return self._ok(
                request,
                {"passed": passed_all, "rules": results},
                "compliance check complete",
            )
        return self._fail(request, f"unsupported action {request.action!r}")

    def _ok(self, request: AgentRequest, output: dict[str, Any], message: str) -> AgentResult:
        return AgentResult(self.role, request.action, True, output, message, request.task_id)

    def _fail(self, request: AgentRequest, message: str) -> AgentResult:
        return AgentResult(self.role, request.action, False, {}, message, request.task_id)
