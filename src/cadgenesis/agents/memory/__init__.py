"""cadgenesis.agents.memory
==========================
Memory agent.

Read/write gateway to the layer-integrated semantic
:class:`~cadgenesis.memory.MemorySystem`, letting the fleet persist and reuse
design knowledge during a session.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.agents.base import AgentRequest, AgentResult
from cadgenesis.agents.infrastructure import AgentBase, Capability


class MemoryAgent(AgentBase):
    """Stores and recalls records in the semantic memory system."""

    role = "memory"
    actions = ("remember", "recall", "forget")
    version = "1.0.0"
    capabilities = (
        Capability("memory.remember", "write a record into a memory pool"),
        Capability("memory.recall", "read a record by pool and key"),
        Capability("memory.forget", "remove a record"),
    )

    def __init__(self, memory: Any = None) -> None:
        super().__init__()
        if memory is None:
            from cadgenesis.memory.memory_system import MemorySystem

            memory = MemorySystem()
        self.memory = memory

    def process(self, request: AgentRequest) -> AgentResult:
        payload = request.payload
        pool = str(payload.get("pool", "project"))
        key = str(payload.get("key", ""))
        try:
            if request.action == "remember":
                if not key:
                    return self._fail(request, "payload requires 'key'")
                content = payload.get("content")
                entry = self.memory.remember(pool, key, content)
                return self._ok(
                    request,
                    {"pool": pool, "key": key, "stored": True},
                    "record stored",
                )
            if request.action == "recall":
                if not key:
                    return self._fail(request, "payload requires 'key'")
                entry = self.memory.recall(pool, key)
                if entry is None:
                    return self._fail(request, f"no record for key {key!r}")
                return self._ok(
                    request,
                    {"pool": pool, "key": key, "content": entry.content, "entry": entry.to_dict()},
                    "record recalled",
                )
            if request.action == "forget":
                if not key:
                    return self._fail(request, "payload requires 'key'")
                removed = self.memory.forget(pool, key)
                return self._ok(
                    request,
                    {"pool": pool, "key": key, "removed": removed},
                    "record removed" if removed else "record absent",
                )
            return self._fail(request, f"unsupported action {request.action!r}")
        except Exception as exc:
            return self._fail(request, f"{type(exc).__name__}: {exc}")

    def _ok(self, request: AgentRequest, output: dict[str, Any], message: str) -> AgentResult:
        return AgentResult(self.role, request.action, True, output, message, request.task_id)

    def _fail(self, request: AgentRequest, message: str) -> AgentResult:
        return AgentResult(self.role, request.action, False, {}, message, request.task_id)
