"""cadgenesis.agents.monitoring
=============================
Monitoring agent.

Aggregates health snapshots and progress telemetry across the fleet using the
:class:`~cadgenesis.agents.health.AgentHealthMonitor`; reports fleet status and
workflow progress.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.agents.base import AgentRequest, AgentResult
from cadgenesis.agents.health import AgentHealthMonitor
from cadgenesis.agents.infrastructure import AgentBase, Capability


class MonitoringAgent(AgentBase):
    """Fleet health and progress monitoring."""

    role = "monitoring"
    actions = ("health", "report")
    version = "1.0.0"
    capabilities = (
        Capability("monitoring.health", "aggregate fleet health"),
        Capability("monitoring.report", "emit a workflow progress report"),
    )

    def __init__(self, monitor: AgentHealthMonitor | None = None) -> None:
        super().__init__()
        self.monitor = monitor or AgentHealthMonitor()

    def process(self, request: AgentRequest) -> AgentResult:
        payload = request.payload
        if request.action == "health":
            agents = payload.get("agents", [])
            statuses = self.monitor.check_all(agents)
            return self._ok(
                request,
                {
                    "summary": self.monitor.summary(agents),
                    "statuses": [s.to_dict() for s in statuses],
                },
                "fleet health aggregated",
            )
        if request.action == "report":
            progress = payload.get("progress", {})
            return self._ok(
                request,
                {
                    "progress": progress,
                    "report": f"workflow progress: {progress}",
                },
                "progress report generated",
            )
        return self._fail(request, f"unsupported action {request.action!r}")

    def _ok(self, request: AgentRequest, output: dict[str, Any], message: str) -> AgentResult:
        return AgentResult(self.role, request.action, True, output, message, request.task_id)

    def _fail(self, request: AgentRequest, message: str) -> AgentResult:
        return AgentResult(self.role, request.action, False, {}, message, request.task_id)
