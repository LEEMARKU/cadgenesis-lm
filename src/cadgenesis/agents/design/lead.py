"""cadgenesis.agents.design.lead
===============================
Lead architect agent for the design swarm.

:class:`LeadArchitectAgent` is the director of the autonomous design loop.
Its ``design`` action runs the full stress -> reinforce -> DFM -> cost
workflow and returns a :class:`~cadgenesis.agents.design.loop.DesignReport`;
its ``iterate`` action advances a persisted design state by exactly one
cycle so external coordinators can steer the process step by step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cadgenesis.agents.base import AgentRequest, AgentResult
from cadgenesis.agents.infrastructure import AgentBase, Capability

if TYPE_CHECKING:
    from cadgenesis.agents.design.loop import DesignOrchestrationLoop


class LeadArchitectAgent(AgentBase):
    """Runs the autonomous multi-agent design-orchestration loop."""

    role = "lead_architect"
    actions = ("design", "iterate")
    version = "1.0.0"
    capabilities = (
        Capability(
            "design.orchestrate",
            "run the full stress/DFM/cost design loop until convergence",
            inputs=("part", "load_cases", "target_safety_factor"),
            outputs=("converged", "iterations", "final_parameters"),
        ),
        Capability(
            "design.iterate",
            "advance a design state by one reinforcement/DFM/cost cycle",
            inputs=("state",),
            outputs=("passed", "retryable", "state"),
        ),
    )

    def __init__(self, loop: DesignOrchestrationLoop | None = None) -> None:
        super().__init__()
        if loop is None:
            from cadgenesis.agents.design.loop import DesignOrchestrationLoop

            loop = DesignOrchestrationLoop()
        self.loop = loop

    def process(self, request: AgentRequest) -> AgentResult:
        try:
            if request.action == "design":
                return self._design(request)
            if request.action == "iterate":
                return self._iterate(request)
            return self._fail(request, f"unsupported action {request.action!r}")
        except (KeyError, TypeError, ValueError) as exc:
            return self._fail(request, f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- design

    def _design(self, request: AgentRequest) -> AgentResult:
        task = request.payload
        if "part" not in task:
            return self._fail(request, "design requires a task with a 'part'")
        report = self.loop.run(task)
        return AgentResult(
            self.role,
            request.action,
            ok=report.converged,
            output=report.to_dict(),
            message=(
                f"design converged after {len(report.iterations)} iteration(s)"
                if report.converged
                else "design did not converge"
            ),
            task_id=request.task_id,
        )

    # --------------------------------------------------------------- iterate

    def _iterate(self, request: AgentRequest) -> AgentResult:
        state = request.payload.get("state")
        if not isinstance(state, dict) or "part" not in state:
            return self._fail(request, "iterate requires a 'state' dict with a 'part'")
        iteration = self.loop.step(state)
        return AgentResult(
            self.role,
            request.action,
            ok=True,
            output=iteration.to_dict() | {"state": iteration.state},
            message=iteration.message,
            task_id=request.task_id,
        )

    # ----------------------------------------------------------------- misc

    def _fail(self, request: AgentRequest, message: str) -> AgentResult:
        return AgentResult(self.role, request.action, False, {}, message, request.task_id)


__all__ = ["LeadArchitectAgent"]
