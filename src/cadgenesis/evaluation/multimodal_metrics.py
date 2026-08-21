"""cadgenesis.evaluation.multimodal_metrics
==========================================
Multimodal understanding metrics (Pillar 3).

Measures the quality of the shared engineering embedding space:

* **Retrieval metrics** — recall@k and mean reciprocal rank of
  cross-modal retrieval (text->CAD, CAD->text, ...).
* **Alignment metrics** — average within-pair cosine similarity vs.
  cross-pair similarity (the "alignment gap").
* **Fusion metrics** — intra-class vs. inter-class distance ratio of the
  fused representation.
* **Search benchmark** — a small end-to-end retrieval harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class MultimodalMetrics:
    """Aggregated multimodal metrics."""

    recall_at_k: dict[str, float] = field(default_factory=dict)
    mean_reciprocal_rank: float = 0.0
    alignment: float = 0.0
    alignment_gap: float = 0.0
    fusion_intra_inter_ratio: float = 0.0
    score: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "recall_at_k": self.recall_at_k,
            "mrr": self.mean_reciprocal_rank,
            "alignment": self.alignment,
            "alignment_gap": self.alignment_gap,
            "fusion_intra_inter_ratio": self.fusion_intra_inter_ratio,
            "score": self.score,
        }


def recall_at_k(similarity: torch.Tensor, k: int = 1) -> float:
    """Fraction of rows where the diagonal entry is in the top-k.

    ``similarity`` is an ``(N, N)`` matrix; row ``i`` must retrieve column
    ``i`` among its ``k`` most similar entries (ties broken deterministically).
    """
    n = similarity.shape[0]
    if n == 0:
        return 0.0
    k = max(1, min(k, n))
    top = similarity.topk(k, dim=1).indices
    diagonal = torch.arange(n, device=similarity.device)[:, None]
    hits = (top == diagonal).any(dim=1).sum().item()
    return hits / n


def mean_reciprocal_rank(similarity: torch.Tensor) -> float:
    """MRR over an ``(N, N)`` retrieval matrix (diagonal is the target)."""
    n = similarity.shape[0]
    if n == 0:
        return 0.0
    ranks = torch.argsort(similarity, dim=1, descending=True)
    diagonal = torch.arange(n, device=similarity.device)[:, None]
    positions = (ranks == diagonal).nonzero(as_tuple=True)[1].float()
    return (1.0 / (positions + 1.0)).mean().item()


def cross_modal_retrieval(
    query: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Cosine-similarity retrieval matrix ``(N, N)`` between two batches."""
    q = F.normalize(query, p=2, dim=-1)
    t = F.normalize(target, p=2, dim=-1)
    return q @ t.transpose(-1, -2)


def alignment_metrics(
    query: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float]:
    """Within-pair alignment and the alignment gap.

    Returns ``(alignment, gap)`` where ``alignment`` is the mean diagonal
    cosine similarity and ``gap`` is the mean off-diagonal similarity.  A
    well-aligned shared space has high ``alignment`` and a large gap.
    """
    sim = cross_modal_retrieval(query, target)
    n = sim.shape[0]
    if n == 0:
        return 0.0, 0.0
    on_diag = torch.diagonal(sim).mean().item()
    mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    off_diag = sim[mask].mean().item()
    return on_diag, on_diag - off_diag


def fusion_intra_inter_ratio(
    fused: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Ratio of mean intra-class distance to mean inter-class distance.

    Values << 1 mean the fused representation clusters well by class.
    """
    if fused.shape[0] < 2:
        return 1.0
    normalized = F.normalize(fused, p=2, dim=-1)
    distances = torch.cdist(normalized, normalized, p=2)
    n = distances.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=distances.device)
    same = labels[:, None] == labels[None, :]
    same = same & mask
    diff = mask & ~(labels[:, None] == labels[None, :])
    intra = distances[same].mean().item() if same.any() else 0.0
    inter = distances[diff].mean().item() if diff.any() else 1.0
    if inter <= 0:
        return 1.0
    return intra / inter


def evaluate_retrieval(
    query_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    top_k: int = 1,
) -> MultimodalMetrics:
    """Run the full retrieval + alignment evaluation suite."""
    similarity = cross_modal_retrieval(query_embeddings, target_embeddings)
    alignment, gap = alignment_metrics(query_embeddings, target_embeddings)
    mrr = mean_reciprocal_rank(similarity)
    recall = {f"r@{k}": recall_at_k(similarity, k) for k in (1, 3, 5, 10)}
    score = 0.5 * recall["r@1"] + 0.3 * alignment + 0.2 * max(gap, 0.0)
    return MultimodalMetrics(
        recall_at_k=recall,
        mean_reciprocal_rank=mrr,
        alignment=alignment,
        alignment_gap=gap,
        score=score,
    )


def run_retrieval_benchmark(
    system: Any,
    queries: dict[str, list[Any]],
    targets: dict[str, list[Any]],
    top_k: int = 1,
) -> MultimodalMetrics:
    """End-to-end retrieval benchmark through a ``MultimodalSystem``.

    ``queries``/``targets`` map modality names to aligned input lists;
    retrieval is scored modality-pair-wise and averaged.
    """
    results: list[MultimodalMetrics] = []
    names = list(queries)
    for query_name in names:
        query_modality = query_name
        q_emb = system.embed_modality(_modality(query_modality), queries[query_name])
        for target_name, target_inputs in targets.items():
            if target_name == query_name:
                continue
            t_emb = system.embed_modality(_modality(target_name), target_inputs)
            results.append(evaluate_retrieval(q_emb, t_emb, top_k))
    if not results:
        raise ValueError("retrieval benchmark requires at least one pair")
    return _average(results)


def _modality(name: str) -> Any:
    from cadgenesis.multimodal.common import modality_from_name

    return modality_from_name(name)


def _average(results: list[MultimodalMetrics]) -> MultimodalMetrics:
    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return MultimodalMetrics(
        recall_at_k={
            key: mean([r.recall_at_k[key] for r in results]) for key in results[0].recall_at_k
        },
        mean_reciprocal_rank=mean([r.mean_reciprocal_rank for r in results]),
        alignment=mean([r.alignment for r in results]),
        alignment_gap=mean([r.alignment_gap for r in results]),
        fusion_intra_inter_ratio=mean([r.fusion_intra_inter_ratio for r in results]),
        score=mean([r.score for r in results]),
    )


__all__ = [
    "MultimodalMetrics",
    "alignment_metrics",
    "cross_modal_retrieval",
    "evaluate_retrieval",
    "fusion_intra_inter_ratio",
    "mean_reciprocal_rank",
    "recall_at_k",
    "run_retrieval_benchmark",
]
