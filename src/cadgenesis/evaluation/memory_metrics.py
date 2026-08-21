"""cadgenesis.evaluation.memory_metrics
======================================
Memory system metrics (Pillar 6).

Pure-function metrics over retrieval quality, routing accuracy, consolidation
and compression of the semantic memory layer.  They take plain data
(relevant/retrieved keys, pool predictions, sizes) so any benchmark or test
can plug in concrete values.
"""

from __future__ import annotations

from typing import Any


def precision_at_k(relevant: list[str], retrieved: list[str], k: int | None = None) -> float:
    """Fraction of the top-k retrieved keys that are relevant."""
    top = retrieved if k is None else retrieved[:k]
    if not top:
        return 0.0
    return len(set(top) & set(relevant)) / len(top)


def recall_at_k(relevant: list[str], retrieved: list[str], k: int | None = None) -> float:
    """Fraction of all relevant keys captured within the top-k results."""
    if not relevant:
        return 0.0
    top = set(retrieved if k is None else retrieved[:k])
    return len(top & set(relevant)) / len(relevant)


def mean_reciprocal_rank(relevant: list[str], retrieved: list[str]) -> float:
    """Reciprocal rank of the first relevant hit (1.0 when none retrieved)."""
    relevant_set = set(relevant)
    for index, key in enumerate(retrieved, start=1):
        if key in relevant_set:
            return 1.0 / index
    return 0.0


def routing_accuracy(
    predicted_pools: list[str],
    true_pools: list[str],
) -> float:
    """Fraction of queries routed to the correct pool."""
    if not predicted_pools:
        return 0.0
    pairs = zip(predicted_pools, true_pools, strict=True)
    return sum(1.0 for p, t in pairs if p == t) / len(predicted_pools)


def memory_hit_rate(
    stored_keys: list[str],
    retrievable_keys: list[str],
) -> float:
    """Fraction of stored keys that can be retrieved back."""
    if not stored_keys:
        return 0.0
    return len(set(stored_keys) & set(retrievable_keys)) / len(stored_keys)


def consolidation_ratio(consumed: int, created: int) -> float:
    """Knowledge folded per consolidation pass (created / consumed)."""
    if consumed == 0:
        return 0.0
    return created / consumed


def compression_ratio(original_size: int, compressed_size: int) -> float:
    """Space reduction factor: ``original / compressed`` (>=1 means smaller)."""
    if compressed_size <= 0:
        return 0.0
    return original_size / compressed_size


def run_memory_benchmark(
    retrieval_batches: list[tuple[list[str], list[str]]],
    routing_preds: list[tuple[str, str]] | None = None,
    consolidation_batch: tuple[int, int] = (0, 0),
    compression_batch: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    """Aggregate the standard memory metrics into one report."""
    retrieved_results: dict[str, float] = {
        "p@5": 0.0,
        "r@5": 0.0,
        "mrr": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    if retrieval_batches:
        p_sum = r_sum = mrr_sum = 0.0
        for relevant, retrieved in retrieval_batches:
            p_sum += precision_at_k(relevant, retrieved)
            r_sum += recall_at_k(relevant, retrieved)
            mrr_sum += mean_reciprocal_rank(relevant, retrieved)
        count = len(retrieval_batches)
        retrieved_results = {
            "precision": p_sum / count,
            "recall": r_sum / count,
            "mrr": mrr_sum / count,
            "p@5": precision_at_k(
                [k for rel, _ in retrieval_batches for k in rel],
                [k for _, ret in retrieval_batches for k in ret],
                k=5,
            ),
            "r@5": recall_at_k(
                [k for rel, _ in retrieval_batches for k in rel],
                [k for _, ret in retrieval_batches for k in ret],
                k=5,
            ),
        }
    consumed, created = consolidation_batch
    report: dict[str, Any] = {
        "retrieval": retrieved_results,
        "consolidation_ratio": consolidation_ratio(consumed, created),
        "compression_ratio": compression_ratio(*compression_batch),
        "checks": {
            "retrieval_batches": len(retrieval_batches),
            "routing_queries": len(routing_preds or []),
        },
    }
    if routing_preds:
        predicted = [p for p, _ in routing_preds]
        true = [t for _, t in routing_preds]
        report["routing_accuracy"] = routing_accuracy(predicted, true)
    else:
        report["routing_accuracy"] = 0.0
    return report


__all__ = [
    "compression_ratio",
    "consolidation_ratio",
    "mean_reciprocal_rank",
    "memory_hit_rate",
    "precision_at_k",
    "recall_at_k",
    "routing_accuracy",
    "run_memory_benchmark",
]
