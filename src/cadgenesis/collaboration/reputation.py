"""
cadgenesis.collaboration.reputation
===================================
Reputation System for the CADGenesis-LM Collaborative Research Economy.

Measures contribution quality, review quality, benchmark performance,
and reliability to compute a comprehensive reputation score.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cadgenesis.collaboration.contributors import ContributorRegistry

logger = logging.getLogger("cadgenesis.collaboration.reputation")


@dataclass
class ReputationComponent:
    """A single component of the reputation score."""

    name: str
    score: float  # 0.0 to 1.0
    weight: float
    details: dict[str, Any] = field(default_factory=dict)

    def weighted_score(self) -> float:
        return self.score * self.weight

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReputationScore:
    """Complete reputation score for a contributor."""

    contributor_id: str
    overall_score: float  # 0.0 to 1.0
    components: list[ReputationComponent]
    tier: str  # "newcomer", "contributor", "expert", "maintainer", "core"
    percentile: float  # 0.0 to 100.0
    computed_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contributor_id": self.contributor_id,
            "overall_score": self.overall_score,
            "components": [c.to_dict() for c in self.components],
            "tier": self.tier,
            "percentile": self.percentile,
            "computed_at": self.computed_at,
            "metadata": self.metadata,
        }


# Tier thresholds
TIER_THRESHOLDS = {
    "newcomer": 0.0,
    "contributor": 0.2,
    "expert": 0.4,
    "maintainer": 0.6,
    "core": 0.8,
}

# Default weights for reputation components
DEFAULT_WEIGHTS = {
    "contribution_quality": 0.35,
    "review_quality": 0.20,
    "benchmark_performance": 0.20,
    "reliability": 0.15,
    "longevity": 0.10,
}


class ReputationEngine:
    """
    Computes and manages reputation scores for contributors.

    Components:
    - Contribution Quality: Impact scores of verified contributions
    - Review Quality: Quality and helpfulness of reviews performed
    - Benchmark Performance: Rankings in benchmark submissions
    - Reliability: Consistency and follow-through on commitments
    - Longevity: Time active in the ecosystem
    """

    def __init__(
        self,
        contributor_registry: ContributorRegistry,
        weights: Mapping[str, float] | None = None,
        directory: str | os.PathLike[str] | None = None,
    ) -> None:
        self.contributor_registry = contributor_registry
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self._lock = threading.RLock()
        self._cache: dict[str, ReputationScore] = {}
        self._cache_ttl = 3600  # 1 hour
        self._last_computed: dict[str, float] = {}

        # Persistence
        self.directory = Path(directory) if directory else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    def _compute_tier(self, score: float) -> str:
        """Determine tier from overall score."""
        for tier in ["core", "maintainer", "expert", "contributor", "newcomer"]:
            if score >= TIER_THRESHOLDS[tier]:
                return tier
        return "newcomer"

    def _compute_percentile(self, score: float, all_scores: list[float]) -> float:
        """Compute percentile rank among all contributors."""
        if not all_scores:
            return 50.0
        below = sum(1 for s in all_scores if s < score)
        return (below / len(all_scores)) * 100.0

    def _calculate_contribution_quality(self, contributor_id: str) -> ReputationComponent:
        """Calculate contribution quality component."""
        contributions = self.contributor_registry.get_contributions(contributor_id)
        verified = [c for c in contributions if c.verified]

        if not verified:
            return ReputationComponent(
                name="contribution_quality",
                score=0.0,
                weight=self.weights["contribution_quality"],
                details={"total": len(contributions), "verified": 0},
            )

        # Average impact score of verified contributions
        avg_impact = sum(c.impact_score for c in verified) / len(verified)
        # Normalize: assume max impact is 10.0, use sigmoid
        normalized = 1.0 - math.exp(-avg_impact / 5.0)

        return ReputationComponent(
            name="contribution_quality",
            score=min(1.0, normalized),
            weight=self.weights["contribution_quality"],
            details={
                "total_contributions": len(contributions),
                "verified_contributions": len(verified),
                "avg_impact_score": round(avg_impact, 3),
            },
        )

    def _calculate_review_quality(self, contributor_id: str) -> ReputationComponent:
        """Calculate review quality component."""
        reviews = self.contributor_registry.get_contributions(
            contributor_id, contribution_type="review"
        )
        verified_reviews = [r for r in reviews if r.verified]

        if not verified_reviews:
            return ReputationComponent(
                name="review_quality",
                score=0.0,
                weight=self.weights["review_quality"],
                details={"total_reviews": len(reviews), "verified_reviews": 0},
            )

        # Average impact score as proxy for review quality
        avg_quality = sum(r.impact_score for r in verified_reviews) / len(verified_reviews)
        normalized = 1.0 - math.exp(-avg_quality / 3.0)

        return ReputationComponent(
            name="review_quality",
            score=min(1.0, normalized),
            weight=self.weights["review_quality"],
            details={
                "total_reviews": len(reviews),
                "verified_reviews": len(verified_reviews),
                "avg_quality_score": round(avg_quality, 3),
            },
        )

    def _calculate_benchmark_performance(self, contributor_id: str) -> ReputationComponent:
        """Calculate benchmark performance component."""
        benchmarks = self.contributor_registry.get_contributions(
            contributor_id, contribution_type="benchmark"
        )
        verified_benchmarks = [b for b in benchmarks if b.verified]

        if not verified_benchmarks:
            return ReputationComponent(
                name="benchmark_performance",
                score=0.0,
                weight=self.weights["benchmark_performance"],
                details={"total_benchmarks": len(benchmarks), "verified_benchmarks": 0},
            )

        # Use impact score as proxy for benchmark ranking
        avg_performance = sum(b.impact_score for b in verified_benchmarks) / len(
            verified_benchmarks
        )
        normalized = 1.0 - math.exp(-avg_performance / 4.0)

        return ReputationComponent(
            name="benchmark_performance",
            score=min(1.0, normalized),
            weight=self.weights["benchmark_performance"],
            details={
                "total_benchmarks": len(benchmarks),
                "verified_benchmarks": len(verified_benchmarks),
                "avg_performance_score": round(avg_performance, 3),
            },
        )

    def _calculate_reliability(self, contributor_id: str) -> ReputationComponent:
        """Calculate reliability component based on consistency and follow-through."""
        contributions = self.contributor_registry.get_contributions(contributor_id)

        if not contributions:
            return ReputationComponent(
                name="reliability",
                score=0.5,  # Neutral for new contributors
                weight=self.weights["reliability"],
                details={"total_contributions": 0},
            )

        # Factors:
        # - Verification rate (how often contributions get verified)
        # - Consistency (regular activity over time)
        verified_count = sum(1 for c in contributions if c.verified)
        verification_rate = verified_count / len(contributions) if contributions else 0

        # Consistency: check activity over last 90 days vs older
        now = time.time()
        recent_cutoff = now - (90 * 86400)
        recent = [c for c in contributions if c.created_at >= recent_cutoff]
        older = [c for c in contributions if c.created_at < recent_cutoff]

        consistency = 0.5
        if older:
            recent_rate = len(recent) / max(1, (now - recent_cutoff) / 86400)
            older_rate = len(older) / max(
                1, (recent_cutoff - min(c.created_at for c in older)) / 86400
            )
            if older_rate > 0:
                consistency = min(1.0, recent_rate / older_rate)

        score = (verification_rate * 0.6) + (consistency * 0.4)

        return ReputationComponent(
            name="reliability",
            score=score,
            weight=self.weights["reliability"],
            details={
                "total_contributions": len(contributions),
                "verification_rate": round(verification_rate, 3),
                "consistency_score": round(consistency, 3),
                "recent_activity_count": len(recent),
            },
        )

    def _calculate_longevity(self, contributor_id: str) -> ReputationComponent:
        """Calculate longevity component based on time in ecosystem."""
        contributor = self.contributor_registry.get(contributor_id)
        if contributor is None:
            return ReputationComponent(
                name="longevity",
                score=0.0,
                weight=self.weights["longevity"],
                details={},
            )

        days_active = (time.time() - contributor.created_at) / 86400
        # Sigmoid: reaches ~0.9 at 1 year, ~0.99 at 2 years
        score = 1.0 - math.exp(-days_active / 180.0)

        return ReputationComponent(
            name="longevity",
            score=min(1.0, score),
            weight=self.weights["longevity"],
            details={"days_active": round(days_active, 1)},
        )

    def compute_reputation(self, contributor_id: str, force: bool = False) -> ReputationScore:
        """Compute the full reputation score for a contributor."""
        with self._lock:
            # Check cache
            if not force and contributor_id in self._cache:
                last = self._last_computed.get(contributor_id, 0)
                if time.time() - last < self._cache_ttl:
                    return self._cache[contributor_id]

            contributor = self.contributor_registry.get(contributor_id)
            if contributor is None:
                raise KeyError(f"contributor {contributor_id!r} not found")

            # Calculate all components
            components = [
                self._calculate_contribution_quality(contributor_id),
                self._calculate_review_quality(contributor_id),
                self._calculate_benchmark_performance(contributor_id),
                self._calculate_reliability(contributor_id),
                self._calculate_longevity(contributor_id),
            ]

            # Weighted overall score
            overall = sum(c.weighted_score() for c in components)
            tier = self._compute_tier(overall)

            # Compute percentile (need all scores)
            all_contributors = self.contributor_registry.list_contributors(limit=10000)
            all_scores = []
            for c in all_contributors:
                if c.id == contributor_id:
                    continue
                try:
                    other_score = self.compute_reputation(c.id).overall_score
                    all_scores.append(other_score)
                except Exception:
                    pass

            percentile = self._compute_percentile(overall, all_scores)

            reputation = ReputationScore(
                contributor_id=contributor_id,
                overall_score=overall,
                components=components,
                tier=tier,
                percentile=percentile,
                metadata={"weights": dict(self.weights)},
            )

            self._cache[contributor_id] = reputation
            self._last_computed[contributor_id] = time.time()

            # Persist if directory configured
            if self.directory:
                self._persist_reputation(reputation)

            return reputation

    def _persist_reputation(self, reputation: ReputationScore) -> None:
        """Persist a reputation score to disk."""
        if not self.directory:
            return
        path = self.directory / f"reputation_{reputation.contributor_id}.json"
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".reputation-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(reputation.to_dict(), handle, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def get_leaderboard(self, limit: int = 50, tier: str | None = None) -> list[ReputationScore]:
        """Get top contributors by reputation."""
        contributors = self.contributor_registry.list_contributors(limit=1000)
        scores = []
        for c in contributors:
            try:
                rep = self.compute_reputation(c.id)
                if tier is None or rep.tier == tier:
                    scores.append(rep)
            except Exception:
                pass
        scores.sort(key=lambda s: s.overall_score, reverse=True)
        return scores[:limit]

    def get_tier_distribution(self) -> dict[str, int]:
        """Get count of contributors per tier."""
        contributors = self.contributor_registry.list_contributors(limit=10000)
        distribution = {tier: 0 for tier in TIER_THRESHOLDS}
        for c in contributors:
            try:
                rep = self.compute_reputation(c.id)
                distribution[rep.tier] = distribution.get(rep.tier, 0) + 1
            except Exception:
                distribution["newcomer"] = distribution.get("newcomer", 0) + 1
        return distribution

    def invalidate_cache(self, contributor_id: str | None = None) -> None:
        """Invalidate cached reputation scores."""
        with self._lock:
            if contributor_id:
                self._cache.pop(contributor_id, None)
                self._last_computed.pop(contributor_id, None)
            else:
                self._cache.clear()
                self._last_computed.clear()


__all__ = [
    "DEFAULT_WEIGHTS",
    "TIER_THRESHOLDS",
    "ReputationComponent",
    "ReputationEngine",
    "ReputationScore",
]
