"""cadgenesis.agents.learning
===========================
Learning agent.

Records lessons, successful solutions and feedback into the memory system and
optionally into the continual-learning substrate, closing the loop between
execution and improvement.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.agents.base import AgentRequest, AgentResult
from cadgenesis.agents.infrastructure import AgentBase, Capability


class LearningAgent(AgentBase):
    """Captures and surfaces learned experiences from the fleet."""

    role = "learning"
    actions = ("record_experience", "recall_lessons", "suggest")
    version = "1.0.0"
    capabilities = (
        Capability("learning.record", "store a lesson or reusable solution"),
        Capability("learning.recall", "recall previously learned lessons"),
        Capability("learning.suggest", "suggest the best learned solution for a task"),
    )

    def __init__(self, memory: Any = None, pool: str = "engineering") -> None:
        super().__init__()
        if memory is None:
            from cadgenesis.memory.memory_system import MemorySystem

            memory = MemorySystem()
        self.memory = memory
        self.pool = pool

    def process(self, request: AgentRequest) -> AgentResult:
        payload = request.payload
        try:
            if request.action == "record_experience":
                lesson = payload.get("lesson")
                if lesson is None:
                    return self._fail(request, "payload requires 'lesson'")
                task = str(payload.get("task", "general"))
                outcome = payload.get("outcome")
                self.memory.remember(
                    self.pool,
                    f"lesson:{task}",
                    {"lesson": lesson, "outcome": outcome, "task": task},
                )
                return self._ok(request, {"task": task, "stored": True}, "experience recorded")
            if request.action == "recall_lessons":
                task = str(payload.get("task", ""))
                query = task or "lesson"
                result = self.memory.retrieve(query, top_k=int(payload.get("top_k", 5)))
                lessons = [
                    {
                        "task": hit.entry.metadata.get("task", ""),
                        "lesson": hit.entry.content.get("lesson", hit.entry.content),
                    }
                    for hit in result.hits
                ]
                return self._ok(
                    request, {"lessons": lessons, "count": len(lessons)}, "lessons recalled"
                )
            if request.action == "suggest":
                task = str(payload.get("task", ""))
                result = self.memory.retrieve(task or "solution", top_k=1, pool_names=(self.pool,))
                if not result.hits:
                    return self._ok(request, {"suggestion": None}, "no learned solution yet")
                return self._ok(
                    request,
                    {"suggestion": result.hits[0].entry.content, "score": result.hits[0].score},
                    "suggestion provided",
                )
            return self._fail(request, f"unsupported action {request.action!r}")
        except Exception as exc:
            return self._fail(request, f"{type(exc).__name__}: {exc}")

    def _ok(self, request: AgentRequest, output: dict[str, Any], message: str) -> AgentResult:
        return AgentResult(self.role, request.action, True, output, message, request.task_id)

    def _fail(self, request: AgentRequest, message: str) -> AgentResult:
        return AgentResult(self.role, request.action, False, {}, message, request.task_id)
