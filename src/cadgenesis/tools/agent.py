"""cadgenesis.tools.agent
=======================
Agent-side bridge: turns :class:`AgentRequest` payloads into validated
tool calls on the :class:`ToolExecutor`, speaking the agent
``AgentRequest`` / ``AgentResult`` envelope.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.agents.base import AgentRequest, AgentResult
from cadgenesis.tools.executor import ToolExecutor
from cadgenesis.tools.schema import Permission, ToolCall


class AgentToolBridge:
    """Adapter so any role agent can emit tool calls.

    Payload contract::

        {"tool": "execute_program",
         "arguments": {"program": ["BOX", "NUM_80", ...]},
         "permission": "execute"}   # optional, default "execute"
    """

    ACTION = "tool_call"

    def __init__(self, executor: ToolExecutor | None = None) -> None:
        self.executor = executor or ToolExecutor()

    def can_handle(self, action: str) -> bool:
        return action == self.ACTION

    def handle(self, request: AgentRequest) -> AgentResult:
        """Dispatch ``request.payload`` as a tool call."""
        tool_name = request.payload.get("tool")
        if not isinstance(tool_name, str) or not tool_name:
            return AgentResult(
                role=request.role,
                action=request.action,
                ok=False,
                message="payload requires a 'tool' name",
                task_id=request.task_id,
            )
        try:
            granted = Permission(request.payload.get("permission", "execute"))
        except ValueError:
            granted = Permission.EXECUTE
        call = ToolCall(
            name=tool_name,
            arguments=request.payload.get("arguments", {}),
            caller=request.role,
        )
        result = self.executor.dispatch(call, granted=granted)
        return AgentResult(
            role=request.role,
            action=request.action,
            ok=result.ok,
            output={"tool": tool_name, "output": result.output},
            message=result.error or f"tool {tool_name!r} succeeded",
            task_id=request.task_id,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "action": self.ACTION,
            "tools": self.executor.registry.list_tools(),
        }


__all__ = ["AgentToolBridge"]
