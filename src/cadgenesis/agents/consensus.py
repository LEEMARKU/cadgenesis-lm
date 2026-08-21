"""cadgenesis.agents.consensus
============================
Consensus engine for the multi-agent orchestration layer.

When several agents return opinions on the same question, the engine combines
them into a single decision: majority vote, weighted vote, numeric mean, or a
confidence-weighted ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentOpinion:
    """A single agent's answer to a consensus question."""

    agent: str
    option: Any
    weight: float = 1.0
    confidence: float = 1.0


class ConsensusEngine:
    """Aggregates opinions from multiple agents into a decision.

    Pillar 5 additions (all optional / backward compatible): a quorum
    requirement, explicit tie-break policy, veto agents, a fallback decision
    when no decision is possible, and a decision trace.
    """

    def __init__(
        self,
        quorum: int = 0,
        tie_break: str = "first",
        fallback: Any = None,
        veto_agents: tuple[str, ...] = (),
    ) -> None:
        if quorum < 0:
            raise ValueError("quorum must be >= 0")
        if tie_break not in ("first", "weighted"):
            raise ValueError("tie_break must be 'first' or 'weighted'")
        self._quorum = quorum
        self._tie_break = tie_break
        self._fallback = fallback
        self._veto_agents = tuple(veto_agents)
        self._opinions: list[AgentOpinion] = []
        self._trace: list[dict[str, Any]] = []

    # ---------------------------------------------------------------- record

    def record(self, opinion: AgentOpinion) -> None:
        if not opinion.agent:
            raise ValueError("opinion requires an agent name")
        self._opinions.append(opinion)

    def record_many(self, opinions: list[AgentOpinion]) -> None:
        for opinion in opinions:
            self.record(opinion)

    @property
    def opinions(self) -> list[AgentOpinion]:
        return list(self._opinions)

    def clear(self) -> None:
        self._opinions.clear()
        self._trace.clear()

    @property
    def count(self) -> int:
        return len(self._opinions)

    # ------------------------------------------------------------ P5 features

    def has_quorum(self) -> bool:
        """True when enough opinions have been recorded to decide."""
        return self.count >= self._quorum

    def required_opinions(self) -> int:
        return self._quorum

    def vetoed(self) -> bool:
        """True when any veto agent has recorded an opinion."""
        return any(op.agent in self._veto_agents for op in self._opinions)

    def _weights(self) -> dict[Any, float]:
        tally: dict[Any, float] = {}
        for opinion in self._opinions:
            tally[opinion.option] = tally.get(opinion.option, 0.0) + (
                opinion.weight * opinion.confidence
            )
        return tally

    def decision(self) -> Any:
        """Return the decided option under the configured policy.

        Order: quorum check → veto check → majority (with tie-break policy).
        Returns ``fallback`` when no decision can be made.
        """
        record: dict[str, Any] = {"options": [op.option for op in self._opinions]}
        if not self.has_quorum() or self.vetoed():
            record["decision"] = self._fallback
            record["method"] = "fallback"
            self._trace.append(record)
            return self._fallback
        option = self.majority()
        record["decision"] = option
        record["method"] = "majority"
        self._trace.append(record)
        return option

    @property
    def trace(self) -> list[dict[str, Any]]:
        return list(self._trace)

    # ------------------------------------------------------------- decisions

    def majority(self) -> Any | None:
        """The option with the most votes.

        Ties are resolved by ``tie_break`` (default ``"first"`` preserves the
        legacy first-recorded behavior; ``"weighted"`` breaks ties by total
        ``weight * confidence``).
        """
        if not self._opinions:
            return None
        tally: dict[Any, int] = {}
        for opinion in self._opinions:
            tally[opinion.option] = tally.get(opinion.option, 0) + 1
        best = max(tally, key=lambda opt: (tally[opt], 0))
        if self._tie_break == "weighted":
            weights = self._weights()
            best = max(tally, key=lambda opt: (tally[opt], weights.get(opt, 0.0)))
        return best

    def weighted_majority(self) -> Any | None:
        """The option with the highest total (weight times confidence) score."""
        if not self._opinions:
            return None
        tally: dict[Any, float] = {}
        for opinion in self._opinions:
            score = opinion.weight * opinion.confidence
            tally[opinion.option] = tally.get(opinion.option, 0.0) + score
        return max(tally, key=lambda opt: tally[opt])

    def mean(self) -> float | None:
        """Numeric mean of opinions (weighted); None when not numeric."""
        weighted_sum = 0.0
        weight_total = 0.0
        for opinion in self._opinions:
            if not isinstance(opinion.option, (int, float)):
                return None
            weight = opinion.weight * opinion.confidence
            weighted_sum += opinion.option * weight
            weight_total += weight
        if weight_total <= 0.0:
            return None
        return weighted_sum / weight_total

    def is_unanimous(self) -> bool:
        """True when every opinion agrees on the same option."""
        if not self._opinions:
            return False
        first = self._opinions[0].option
        return all(op.option == first for op in self._opinions)

    def confidence(self) -> float:
        """Mean confidence of all recorded opinions (0 when empty)."""
        if not self._opinions:
            return 0.0
        return sum(op.confidence for op in self._opinions) / len(self._opinions)

    def summary(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "majority": self.majority(),
            "weighted_majority": self.weighted_majority(),
            "mean": self.mean(),
            "unanimous": self.is_unanimous(),
            "confidence": self.confidence(),
        }

    def full_summary(self) -> dict[str, Any]:
        """Extended summary including the Pillar 5 decision features."""
        data = self.summary()
        data.update(
            {
                "quorum": self._quorum,
                "has_quorum": self.has_quorum(),
                "vetoed": self.vetoed(),
                "tie_break": self._tie_break,
                "decision": self.decision(),
                "trace": self._trace,
            }
        )
        return data
