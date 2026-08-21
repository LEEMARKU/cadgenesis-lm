"""cadgenesis.agents.debugging
============================
Debugging agent.

Inspects failed results and errors across the fleet, groups them by cause and
proposes concrete fixes for the orchestrator to retry.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from cadgenesis.agents.base import AgentRequest, AgentResult
from cadgenesis.agents.infrastructure import AgentBase, Capability

_FIX_HINTS = {
    "cannot handle action": "route the request to an agent that declares the action",
    "requires": "add the missing required payload field",
    "KeyError": "check the referenced key exists in the target store",
    "infeasible": "relax the constraints or widen bounds",
    "not found": "verify the name against the registry/catalog",
    "unknown": "use one of the enumerated allowed values",
}


class DebuggingAgent(AgentBase):
    """Diagnoses failures and proposes fixes."""

    role = "debugging"
    actions = ("inspect", "suggest_fix")
    version = "1.0.0"
    capabilities = (
        Capability("debug.inspect", "classify a batch of results for failures"),
        Capability("debug.fix", "suggest a fix for a failure message"),
    )

    def process(self, request: AgentRequest) -> AgentResult:
        payload = request.payload
        if request.action == "inspect":
            results = payload.get("results", [])
            failed = [r for r in results if getattr(r, "ok", True) is False]
            causes: Counter[str] = Counter()
            details = []
            for result in failed:
                message = getattr(result, "message", "") or ""
                cause = self._classify(message)
                causes[cause] += 1
                details.append(
                    {
                        "role": getattr(result, "role", ""),
                        "action": getattr(result, "action", ""),
                        "cause": cause,
                        "message": message,
                    }
                )
            return self._ok(
                request,
                {
                    "failed": len(failed),
                    "total": len(results),
                    "causes": dict(causes),
                    "details": details,
                },
                f"inspected {len(results)} results",
            )
        if request.action == "suggest_fix":
            message = str(payload.get("message", ""))
            fix = self._classify(message)
            hint = _FIX_HINTS.get(fix, "review the payload and the backend contract")
            return self._ok(request, {"cause": fix, "suggestion": hint}, "fix suggested")
        return self._fail(request, f"unsupported action {request.action!r}")

    @staticmethod
    def _classify(message: str) -> str:
        for key in _FIX_HINTS:
            if key.lower() in message.lower():
                return key
        return "unknown"

    def _ok(self, request: AgentRequest, output: dict[str, Any], message: str) -> AgentResult:
        return AgentResult(self.role, request.action, True, output, message, request.task_id)

    def _fail(self, request: AgentRequest, message: str) -> AgentResult:
        return AgentResult(self.role, request.action, False, {}, message, request.task_id)
