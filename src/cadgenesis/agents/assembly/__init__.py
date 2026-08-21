"""cadgenesis.agents.assembly
===========================
Specialised assembly agent: mates parts and checks assembly-level geometry.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.agents.base import Agent, AgentRequest, AgentResult
from cadgenesis.reasoning.geometry_reasoner import GeometryReasoner, Primitive


class AssemblyAgent(Agent):
    """Checks part fits, clearances and mates within an assembly."""

    role = "assembly"
    actions = ("check_clearance", "check_mate")

    def __init__(self) -> None:
        super().__init__()
        self.reasoner = GeometryReasoner

    def _primitive(self, payload: dict[str, Any]) -> Primitive:
        return Primitive(
            kind=str(payload["kind"]),
            dims=dict(payload.get("dims", {})),
            position=payload.get("position"),
            name=str(payload.get("name", "")),
        )

    def process(self, request: AgentRequest) -> AgentResult:
        payload = request.payload
        try:
            a = self._primitive(payload["a"])
            b = self._primitive(payload["b"])
            if request.action == "check_clearance":
                gap = float(payload.get("gap", 0.0))
                ok = self.reasoner.check_clearance(a, b, gap)
                clearance = self.reasoner.clearance(a, b)
                return AgentResult(
                    role=self.role,
                    action=request.action,
                    ok=ok,
                    output={"clearance": clearance, "required_gap": gap},
                    message=(
                        "clearance OK" if ok else f"clearance {clearance:.3f} < required {gap:.3f}"
                    ),
                    task_id=request.task_id,
                )
            if request.action == "check_mate":
                # Mating must be interference-free (AABBs must not overlap).
                overlaps = self.reasoner.overlaps(a, b)
                ok = not overlaps
                return AgentResult(
                    role=self.role,
                    action=request.action,
                    ok=ok,
                    output={"overlaps": overlaps, "interference_free": ok},
                    message="mates OK" if ok else "parts overlap; mate invalid",
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
        return AgentResult(
            role=self.role,
            action=request.action,
            ok=False,
            message=f"unsupported action {request.action!r}",
            task_id=request.task_id,
        )
