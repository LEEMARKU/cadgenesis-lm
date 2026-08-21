"""cadgenesis.agents.user
========================
User interaction agent.

Manages user preferences, workflow and design-style memory so the fleet can
personalize behavior, and answers interaction requests from the user.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.agents.base import AgentRequest, AgentResult
from cadgenesis.agents.infrastructure import AgentBase, Capability


class UserInteractionAgent(AgentBase):
    """Gathers and serves user preferences and interaction context."""

    role = "user"
    actions = ("set_preference", "get_preference", "record_style", "ask")
    version = "1.0.0"
    capabilities = (
        Capability("user.preference", "set and read user preferences"),
        Capability("user.style", "record the user's design style"),
        Capability("user.interaction", "answer interaction requests"),
    )

    def __init__(self, user_memory: Any = None) -> None:
        super().__init__()
        if user_memory is None:
            from cadgenesis.memory.user_memory import UserMemory

            user_memory = UserMemory()
        self.user_memory = user_memory

    def process(self, request: AgentRequest) -> AgentResult:
        payload = request.payload
        try:
            if request.action == "set_preference":
                name = str(payload.get("name", ""))
                if not name:
                    return self._fail(request, "payload requires 'name'")
                self.user_memory.set_preference(name, payload.get("value"))
                return self._ok(request, {"name": name, "set": True}, "preference saved")
            if request.action == "get_preference":
                name = str(payload.get("name", ""))
                if not name:
                    return self._fail(request, "payload requires 'name'")
                value = self.user_memory.get_preference(name)
                if value is None:
                    return self._fail(request, f"unknown preference {name!r}")
                return self._ok(request, {"name": name, "value": value}, "preference read")
            if request.action == "record_style":
                style = payload.get("style")
                if style is None:
                    return self._fail(request, "payload requires 'style'")
                if isinstance(style, str):
                    style = {"name": style}
                self.user_memory.record_style(style)
                return self._ok(request, {"style": style, "recorded": True}, "style recorded")
            if request.action == "ask":
                question = str(payload.get("question", ""))
                return self._ok(
                    request,
                    {"question": question, "reply": "Acknowledged. Preferences applied."},
                    "interaction acknowledged",
                )
            return self._fail(request, f"unsupported action {request.action!r}")
        except Exception as exc:
            return self._fail(request, f"{type(exc).__name__}: {exc}")

    def _ok(self, request: AgentRequest, output: dict[str, Any], message: str) -> AgentResult:
        return AgentResult(self.role, request.action, True, output, message, request.task_id)

    def _fail(self, request: AgentRequest, message: str) -> AgentResult:
        return AgentResult(self.role, request.action, False, {}, message, request.task_id)
