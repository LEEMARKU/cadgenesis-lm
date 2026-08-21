"""cadgenesis.adapters.router
==========================
Adapter routing and selection.

Routes a query to the best registered adapter via (a) exact domain match,
(b) Jaccard token-overlap similarity against domain descriptors, or
(c) a configured default. Pure Python — no embeddings required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> frozenset[str]:
    """Lowercase word tokens (letters, digits, underscores)."""
    return frozenset(_WORD_RE.findall(text.lower()))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard similarity of two token sets (0.0 when either is empty)."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


@dataclass(frozen=True)
class RoutingDecision:
    """Which adapter (if any) a query was routed to, and how."""

    adapter_id: str | None
    score: float
    strategy: str  # 'exact' | 'similarity' | 'default'


class AdapterRouter:
    """Deterministic adapter router with exact / similarity / default strategies."""

    def __init__(self, default_adapter_id: str | None = None) -> None:
        self.default_adapter_id = default_adapter_id
        self._adapters: dict[str, list[str]] = {}
        self._descriptions: dict[str, str] = {}
        self._domains: dict[str, list[str]] = {}

    def register(self, adapter_id: str, domains: list[str], description: str = "") -> None:
        """Register an adapter with its domains and a free-text description."""
        if adapter_id in self._adapters:
            raise ValueError(f"adapter {adapter_id!r} is already registered")
        self._adapters[adapter_id] = list(domains)
        self._descriptions[adapter_id] = description
        for domain in domains:
            normalized = domain.strip().lower()
            if normalized:
                self._domains.setdefault(normalized, []).append(adapter_id)

    def route(self, query: str, domain: str | None = None) -> RoutingDecision:
        """Route ``query``; an explicit ``domain`` short-circuits to exact match."""
        if domain is not None:
            matches = self._domains.get(domain.strip().lower(), [])
            if matches:
                return RoutingDecision(adapter_id=min(matches), score=1.0, strategy="exact")

        query_tokens = _tokenize(query)
        best_id: str | None = None
        best_score = 0.0
        for adapter_id in sorted(self._adapters):
            score = _jaccard(query_tokens, self._adapter_tokens(adapter_id))
            if score > best_score:
                best_score = score
                best_id = adapter_id
        if best_id is not None and best_score > 0.0:
            return RoutingDecision(adapter_id=best_id, score=best_score, strategy="similarity")

        return RoutingDecision(adapter_id=self.default_adapter_id, score=0.0, strategy="default")

    def _adapter_tokens(self, adapter_id: str) -> frozenset[str]:
        """Union of the adapter's domain and description tokens."""
        tokens: set[str] = set()
        for domain in self._adapters[adapter_id]:
            tokens |= _tokenize(domain)
        tokens |= _tokenize(self._descriptions.get(adapter_id, ""))
        return frozenset(tokens)
