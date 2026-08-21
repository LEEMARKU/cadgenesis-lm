"""cadgenesis.agents.simulation
=============================
Specialised simulation agent: reasons about load cases and safety factors.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.agents.base import Agent, AgentRequest, AgentResult


class SimulationAgent(Agent):
    """Checks simulation results against safety-factor requirements."""

    role = "simulation"
    actions = ("check_safety", "check_load_case")

    def __init__(self, default_safety_factor: float = 1.5) -> None:
        super().__init__()
        if default_safety_factor <= 0:
            raise ValueError("default_safety_factor must be > 0")
        self.default_safety_factor = default_safety_factor

    def _safety_result(
        self,
        payload: dict[str, Any],
        task_id: str,
    ) -> AgentResult:
        safety = float(payload.get("safety_factor", 0.0))
        required = float(payload.get("required_safety_factor", self.default_safety_factor))
        ok = safety >= required
        return AgentResult(
            role=self.role,
            action="check_safety",
            ok=ok,
            output={
                "safety_factor": safety,
                "required": required,
                "margin": safety - required,
            },
            message=(
                "safety factor adequate"
                if ok
                else f"safety {safety:.2f} below required {required:.2f}"
            ),
            task_id=task_id,
        )

    def process(self, request: AgentRequest) -> AgentResult:
        payload = request.payload
        if request.action == "check_safety":
            return self._safety_result(payload, request.task_id)
        if request.action == "check_load_case":
            loads = payload.get("loads")
            if not isinstance(loads, list) or not loads:
                return AgentResult(
                    role=self.role,
                    action=request.action,
                    ok=False,
                    message="check_load_case requires a non-empty 'loads' list",
                    task_id=request.task_id,
                )
            max_magnitude = max(float(load.get("magnitude", 0.0)) for load in loads)
            limit = float(payload.get("limit", max_magnitude))
            ok = max_magnitude <= limit
            return AgentResult(
                role=self.role,
                action=request.action,
                ok=ok,
                output={
                    "load_count": len(loads),
                    "max_magnitude": max_magnitude,
                    "limit": limit,
                },
                message=(
                    "load case within limits"
                    if ok
                    else f"load {max_magnitude:.2f} exceeds limit {limit:.2f}"
                ),
                task_id=request.task_id,
            )
        return AgentResult(
            role=self.role,
            action=request.action,
            ok=False,
            message=f"unsupported action {request.action!r}",
            task_id=request.task_id,
        )
