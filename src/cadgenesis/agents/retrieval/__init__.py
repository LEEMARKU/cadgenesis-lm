"""cadgenesis.agents.retrieval
============================
Retrieval agent.

Semantic and hybrid retrieval over the layer-integrated memory system for the
rest of the agent fleet.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.agents.base import AgentRequest, AgentResult
from cadgenesis.agents.infrastructure import AgentBase, Capability


class RetrievalAgent(AgentBase):
    """Retrieves relevant knowledge from memory for a query."""

    role = "retrieval"
    actions = ("retrieve", "route")
    version = "1.0.0"
    capabilities = (
        Capability("retrieval.query", "semantic retrieve across memory pools"),
        Capability("retrieval.route", "determine which pool matches a query"),
    )

    def __init__(self, memory: Any = None) -> None:
        super().__init__()
        if memory is None:
            from cadgenesis.memory.memory_system import MemorySystem

            memory = MemorySystem()
        self.memory = memory

    def process(self, request: AgentRequest) -> AgentResult:
        payload = request.payload
        query = str(payload.get("query", ""))
        try:
            if request.action == "retrieve":
                if not query:
                    return self._fail(request, "payload requires 'query'")
                top_k = int(payload.get("top_k", 8))
                pools = payload.get("pool_names")
                result = self.memory.retrieve(query, top_k=top_k, pool_names=pools)
                hits = [
                    {
                        "pool": hit.pool,
                        "key": hit.entry.key,
                        "score": hit.score,
                        "content": hit.entry.content,
                    }
                    for hit in result.hits
                ]
                return self._ok(
                    request,
                    {"query": query, "hits": hits, "count": len(hits)},
                    f"retrieved {len(hits)} records",
                )
            if request.action == "route":
                if not query:
                    return self._fail(request, "payload requires 'query'")
                decisions = [
                    {"pool": d.pool, "score": d.score, "size": d.size}
                    for d in self.memory.route(query)
                ]
                return self._ok(
                    request,
                    {"query": query, "route": decisions},
                    "routing complete",
                )
            return self._fail(request, f"unsupported action {request.action!r}")
        except Exception as exc:
            return self._fail(request, f"{type(exc).__name__}: {exc}")

    def _ok(self, request: AgentRequest, output: dict[str, Any], message: str) -> AgentResult:
        return AgentResult(self.role, request.action, True, output, message, request.task_id)

    def _fail(self, request: AgentRequest, message: str) -> AgentResult:
        return AgentResult(self.role, request.action, False, {}, message, request.task_id)
