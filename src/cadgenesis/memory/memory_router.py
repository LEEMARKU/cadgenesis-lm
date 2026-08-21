"""cadgenesis.memory.memory_router
=================================
Cross-pool memory routing.

Given a query, the router scores each registered pool by how well its records
match (keyword overlap + domain affinity keywords), then returns the ranked
pool list and/or a routed retrieval result.  This lets a caller ask "which
pool holds what I need?" before issuing a full retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cadgenesis.memory.memory_common import MemoryStore, _tokens
from cadgenesis.memory.retrieval import RetrievalHit, RetrievalResult


@dataclass
class RoutingDecision:
    """A pool ranked for a query."""

    pool: str
    score: float
    size: int


class MemoryRouter:
    """Routes queries to the most relevant memory pools."""

    def __init__(
        self,
        stores: list[MemoryStore] | None = None,
        domain_keywords: dict[str, set[str]] | None = None,
        affinity_weight: float = 0.4,
    ):
        self._stores: dict[str, MemoryStore] = {}
        self._keywords: dict[str, set[str]] = domain_keywords or {}
        self.affinity_weight = affinity_weight
        if stores:
            for store in stores:
                self.register(store)

    # ------------------------------------------------------------- registry

    def register(
        self,
        store: MemoryStore,
        keywords: set[str] | None = None,
    ) -> None:
        """Register a pool, optionally with domain-affinity keywords."""
        self._stores[store.name] = store
        if keywords:
            self._keywords[store.name] = set(keywords)

    def unregister(self, name: str) -> bool:
        self._keywords.pop(name, None)
        return self._stores.pop(name, None) is not None

    @property
    def pool_names(self) -> list[str]:
        return sorted(self._stores)

    def stores(self) -> list[MemoryStore]:
        return list(self._stores.values())

    # -------------------------------------------------------------- routing

    def _pool_score(
        self,
        store: MemoryStore,
        query_tokens: set[str],
    ) -> float:
        keyword_hits = 0.0
        affinity = self._keywords.get(store.name, set())
        if query_tokens and affinity:
            keyword_hits = len(query_tokens & affinity) / len(query_tokens)
        elif query_tokens and not affinity:
            # Fall back to content overlap when no affinity keywords exist.
            keyword_hits = len(store.top(top_k=store.capacity or 1)) / max(len(store), 1)
        return self.affinity_weight * keyword_hits

    def route(self, query: str) -> list[RoutingDecision]:
        """Rank pools by affinity + content relevance for the query."""
        query_tokens = _tokens(query)
        decisions: list[RoutingDecision] = []
        for name in self.pool_names:
            store = self._stores[name]
            score = self._pool_score(store, query_tokens)
            if store:
                hit = store.search(query, top_k=1)
                if hit:
                    score += (1.0 - self.affinity_weight) * hit[0].score
            decisions.append(RoutingDecision(pool=name, score=score, size=len(store)))
        decisions.sort(key=lambda d: d.score, reverse=True)
        return decisions

    # --------------------------------------------------- contextual routing

    def route_by_context(self, context: dict[str, Any]) -> list[RoutingDecision]:
        """Rank pools from an arbitrary context dict.

        ``context`` carries explicit signals (``pool`` / ``pools``), an
        optional ``text`` query, and a ``metadata`` map of pool→boost weights.
        The metadata boost is applied on top of the affinity/content score.
        """
        explicit = context.get("pool") or context.get("pools")
        names = [explicit] if isinstance(explicit, str) else list(explicit or [])
        if names:
            decisions = [
                RoutingDecision(pool=name, score=1.0, size=len(self._stores[name]))
                for name in names
                if name in self._stores
            ]
        else:
            decisions = self.route(str(context.get("text", "")))
        boosts = dict(context.get("metadata") or {})
        for decision in decisions:
            decision.score += float(boosts.get(decision.pool, 0.0))
        decisions.sort(key=lambda d: d.score, reverse=True)
        return decisions

    def route_by_task(self, task_type: str) -> list[RoutingDecision]:
        """Route by a task label (``design``, ``analysis``, ``simulation``, ...).

        Task types map onto pool affinity keywords:
        design → cad/engineering, analysis → engineering, simulation →
        simulation, manufacturing → manufacturing, planning → project,
        preferences → user, transcript/notes → session, generic → working.
        """
        mapping: dict[str, set[str]] = {
            "design": {"feature", "brep", "extrude", "part", "assembly"},
            "analysis": {"tolerance", "standard", "material", "iso"},
            "simulation": {"fea", "cfd", "stress", "load", "safety"},
            "manufacturing": {"machining", "tool", "draft", "mold"},
            "planning": {"project", "milestone", "version"},
            "preferences": {"preference", "style", "user", "profile"},
            "transcript": {"session", "toolbar", "ui", "interaction"},
        }
        keywords = mapping.get(task_type, {"context", "active", "current"})
        return self.route(" ".join(keywords))

    def route_by_confidence(
        self,
        query: str,
        confidence: float,
        low_pool: str = "working",
        high_pool: str = "engineering",
    ) -> list[RoutingDecision]:
        """Route by model confidence.

        High confidence (``>= 0.5``) routes to the authoritative knowledge
        pool; low confidence falls back to the short-term working pool.  The
        preferred pool is guaranteed to rank first so the fallback is always
        honoured.
        """
        decisions = self.route(query)
        preferred = high_pool if confidence >= 0.5 else low_pool
        if not decisions:
            return decisions
        max_other = max(
            (d.score for d in decisions if d.pool != preferred),
            default=0.0,
        )
        for decision in decisions:
            if decision.pool == preferred:
                decision.score = max(decision.score, max_other + 0.01)
        decisions.sort(key=lambda d: d.score, reverse=True)
        return decisions

    def route_by_agent(self, agent_role: str) -> list[RoutingDecision]:
        """Route by the requesting agent role.

        Agent names map to the pool whose domain they work in; unknown roles
        fall back to a generic query over the available pools.
        """
        mapping: dict[str, set[str]] = {
            "planner": {"project", "milestone", "version"},
            "geometry": {"feature", "brep", "extrude", "part"},
            "constraint": {"tolerance", "standard", "material"},
            "assembly": {"assembly", "mate", "part", "clearance"},
            "manufacturing": {"machining", "tool", "draft", "mold"},
            "simulation": {"fea", "cfd", "stress", "load", "safety"},
            "optimization": {"material", "cost", "objective"},
            "validation": {"standard", "iso", "tolerance"},
            "material": {"material", "standard", "iso"},
            "cost": {"material", "cost", "process"},
            "memory": {"context", "active", "current"},
            "retrieval": {"context", "active", "current"},
            "user": {"preference", "style", "user", "profile"},
            "learning": {"context", "active", "draft"},
            "monitoring": {"session", "interaction", "context"},
            "debugging": {"session", "context", "interaction"},
        }
        keywords = mapping.get(agent_role, {"context", "active", "current"})
        return self.route(" ".join(keywords))

    def best_pool(self, query: str) -> str | None:
        """Name of the single most relevant pool, or None when empty."""
        ranked = self.route(query)
        if not ranked:
            return None
        best = ranked[0]
        if best.size == 0 and best.score <= 0.0:
            return None
        return best.pool

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        pool_names: list[str] | None = None,
    ) -> RetrievalResult:
        """Route to the best pools and retrieve from them."""
        names = pool_names or [d.pool for d in self.route(query)]
        merged: dict[str, RetrievalHit] = {}
        for name in names:
            store = self._stores.get(name)
            if store is None:
                continue
            for hit in store.search(query, top_k=top_k):
                existing = merged.get(hit.entry.key)
                if existing is None or hit.score > existing.score:
                    merged[hit.entry.key] = RetrievalHit(
                        entry=hit.entry, pool=name, score=hit.score
                    )
        hits = sorted(merged.values(), key=lambda h: h.score, reverse=True)
        return RetrievalResult(query=query, hits=hits[: max(top_k, 0)])

    def summary(self) -> dict[str, Any]:
        return {
            "pools": {name: store.summary() for name, store in self._stores.items()},
            "affinity_keywords": {k: sorted(v) for k, v in self._keywords.items()},
        }
