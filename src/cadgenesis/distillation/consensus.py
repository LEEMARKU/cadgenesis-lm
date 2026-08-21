"""cadgenesis.distillation.consensus
=================================
Teacher-consensus distillation.

Aggregates the raw TOON outputs of multiple teacher models into a single
consensus candidate before distillation.  Two aggregation levels are
provided:

* :meth:`TeacherConsensus.toon_votes` -- string-level voting over complete
  TOON payloads (exact-match votes, optionally weighted by per-teacher
  trust).  Used when teachers return serialized CAD programs.
* :meth:`TeacherConsensus.consensus_logits` -- tensor-level aggregation
  over per-teacher vocab logits ``(K, B, T, V)`` returning the mean logits
  (the distillation target) and a per-position agreement score.

Composition
-----------
Composes with the existing
:class:`~cadgenesis.distillation.distillation_engine.TeacherConsensusEngine`
via :meth:`TeacherConsensus.sequence_consensus`, which delegates token-list
consensus to that engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cadgenesis.distillation.distillation_engine import TeacherConsensusEngine

__all__ = ["ConsensusResult", "TeacherConsensus"]


@dataclass
class ConsensusResult:
    """Outcome of weighted TOON voting across teachers.

    ``consensus_toon`` is the winning TOON string; ``agreement_score`` the
    weighted share of the vote (``[0, 1]``); ``vote_counts`` maps every
    distinct TOON to its raw number of votes; ``winner_teachers`` lists the
    teachers whose output equals the consensus (in stable order).
    """

    consensus_toon: str
    agreement_score: float
    vote_counts: dict[str, int]
    winner_teachers: list[str]


class TeacherConsensus:
    """Weighted consensus aggregation over teacher TOON outputs.

    Parameters
    ----------
    stable_order:
        Optional teacher-name ordering used to break ties ("first-seen"
        wins).  Defaults to the lexicographic order of the output keys,
        which is deterministic.
    """

    def __init__(self, stable_order: list[str] | None = None) -> None:
        self.stable_order = stable_order

    def _teacher_order(self, outputs: dict[str, str]) -> list[str]:
        if self.stable_order is not None:
            return [t for t in self.stable_order if t in outputs]
        return sorted(outputs)

    def toon_votes(
        self,
        outputs: dict[str, str],
        weights: dict[str, float] | None = None,
    ) -> ConsensusResult:
        """Vote over teacher TOON outputs by exact string match.

        Each teacher contributes ``weights[t]`` (default ``1.0`` when
        ``weights`` is None or omits the teacher) to the total score of its
        TOON.  The winning TOON is the one with the highest weighted score;
        ties are broken by first-seen order in the stable teacher order.
        ``agreement_score`` is ``winner_score / total_weight``, so with
        equal weights it reduces to ``max_votes / num_teachers``.

        Returns an empty :class:`ConsensusResult` when ``outputs`` is empty.
        """
        if not outputs:
            return ConsensusResult(
                consensus_toon="", agreement_score=0.0, vote_counts={}, winner_teachers=[]
            )

        order = self._teacher_order(outputs)
        effective: dict[str, float] = {}
        vote_counts: dict[str, int] = {}
        for teacher in order:
            toon = outputs[teacher]
            weight = weights.get(teacher, 1.0) if weights is not None else 1.0
            effective[toon] = effective.get(toon, 0.0) + weight
            vote_counts[toon] = vote_counts.get(toon, 0) + 1

        total_weight = sum(effective.values())
        consensus_toon = max(effective, key=lambda toon: effective[toon])  # first-seen wins ties
        agreement = effective[consensus_toon] / total_weight if total_weight > 0 else 0.0
        winner_teachers = [t for t in order if outputs[t] == consensus_toon]
        return ConsensusResult(
            consensus_toon=consensus_toon,
            agreement_score=agreement,
            vote_counts=vote_counts,
            winner_teachers=winner_teachers,
        )

    def sequence_consensus(self, teacher_outputs: dict[str, list[str]]) -> tuple[list[str], float]:
        """Delegate token-list consensus to :class:`TeacherConsensusEngine`."""
        return TeacherConsensusEngine().compute_consensus(teacher_outputs)

    @staticmethod
    def consensus_logits(teacher_logits: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Aggregate per-teacher vocab logits into a distillation target.

        ``teacher_logits`` has shape ``(K, B, T, V)`` (teachers, batch,
        sequence, vocab).  Returns ``(mean_logits, agreement)`` where
        ``mean_logits`` is the mean over the teacher axis -- the soft
        distillation target -- and ``agreement`` is the mean over all
        positions of the fraction of teachers whose argmax matches the
        argmax of the mean (``[0, 1]``).
        """
        if teacher_logits.ndim != 4:
            raise ValueError(
                f"teacher_logits must be 4-D (K, B, T, V), got shape {tuple(teacher_logits.shape)}"
            )
        mean_logits = teacher_logits.mean(dim=0)
        teacher_tokens = teacher_logits.argmax(dim=-1)  # (K, B, T)
        consensus_tokens = mean_logits.argmax(dim=-1)  # (B, T)
        agreement = float((teacher_tokens == consensus_tokens.unsqueeze(0)).float().mean().item())
        return mean_logits, agreement
