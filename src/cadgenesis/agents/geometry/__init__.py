"""cadgenesis.agents.geometry
===========================
Specialised geometry agent: validates and analyses geometric primitives.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.agents.base import Agent, AgentRequest, AgentResult
from cadgenesis.reasoning.geometry_reasoner import GeometryReasoner, Primitive


class GeometryAgent(Agent):
    """Performs volume / AABB / overlap / fit analysis on primitives."""

    role = "geometry"
    actions = ("validate", "volume", "aabb", "overlap", "fit")

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
            if request.action in ("validate", "volume", "aabb"):
                primitive = self._primitive(payload)
                if request.action == "validate":
                    check = self.reasoner.validate(primitive)
                    return AgentResult(
                        role=self.role,
                        action=request.action,
                        ok=check.valid,
                        output={"valid": check.valid, "messages": check.messages},
                        message="; ".join(check.messages) or "geometry valid",
                        task_id=request.task_id,
                    )
                if request.action == "volume":
                    volume = self.reasoner.volume(primitive)
                    return AgentResult(
                        role=self.role,
                        action=request.action,
                        ok=volume > 0,
                        output={"volume": volume},
                        message=f"volume {volume:.3f}",
                        task_id=request.task_id,
                    )
                lo, hi = self.reasoner.aabb(primitive)
                return AgentResult(
                    role=self.role,
                    action=request.action,
                    ok=True,
                    output={"min": list(lo), "max": list(hi)},
                    message="bounds computed",
                    task_id=request.task_id,
                )
            if request.action in ("overlap", "fit"):
                a = self._primitive(payload["a"])
                b = self._primitive(payload["b"])
                if request.action == "overlap":
                    overlaps = self.reasoner.overlaps(a, b)
                    return AgentResult(
                        role=self.role,
                        action=request.action,
                        ok=not overlaps,
                        output={"overlaps": overlaps},
                        message="overlap detected" if overlaps else "no overlap",
                        task_id=request.task_id,
                    )
                fits = self.reasoner.check_fit(a, b)
                return AgentResult(
                    role=self.role,
                    action=request.action,
                    ok=fits,
                    output={"fits": fits},
                    message="part fits cavity" if fits else "part does not fit",
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
