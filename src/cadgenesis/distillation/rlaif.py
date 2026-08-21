"""cadgenesis.distillation.rlaif
=============================
RLAIF (AI feedback) alignment for distillation.

RLAIF replaces human preference labels with AI-generated ones: teacher
samples are scored by the rule-based critiquer
(:class:`~cadgenesis.distillation.critique.CritiqueEngine`) and the scores
are turned into Bradley-Terry preference pairs.  The distillation stage
then optimizes

    L_BT = -mean(log_sigmoid(r_chosen - r_rejected))

which pushes the reward of the preferred sample above that of the rejected
one (optionally with label smoothing to soften the signal, mirroring the
DPO loss in ``cadgenesis.distillation.dpo``).

Contract notes
--------------
* :meth:`RLAIFEngine.preference_pairs` consumes ``samples`` -- dicts with
  at least ``"toon"`` and ``"score"`` keys -- sorts them by score
  descending and pairs adjacent entries with *strictly* higher score as
  ``(chosen, rejected)``.  Ties are dropped.
* :meth:`RLAIFEngine.reward_from_critiques` maps ``{"toon", "score"}``
  records to a ``toon -> reward`` dict (score is used directly, in
  ``[0, 1]``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

__all__ = ["RLAIFEngine"]


class RLAIFEngine:
    """Turns critique scores into Bradley-Terry preference signals.

    Parameters
    ----------
    label_smoothing:
        Softening applied to the BT loss: ``(1 - s) * -log_sigmoid(d) +
        s * -log_sigmoid(-d)`` with ``s`` in ``[0, 0.5)``.
    """

    def __init__(self, label_smoothing: float = 0.0) -> None:
        if not 0.0 <= label_smoothing < 0.5:
            raise ValueError(f"label_smoothing must be in [0, 0.5), got {label_smoothing}")
        self.label_smoothing = label_smoothing

    def preference_pairs(
        self, samples: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Build ``(chosen, rejected)`` pairs from critique-scored samples.

        Samples without a ``"toon"`` or numeric ``"score"`` key are
        skipped.  The remainder are sorted by score descending and adjacent
        entries with strictly decreasing score form a pair; equal-score
        neighbours are dropped.
        """
        scored = [
            sample
            for sample in samples
            if sample.get("toon") is not None and sample.get("score") is not None
        ]
        scored.sort(key=lambda sample: float(sample["score"]), reverse=True)
        return [
            (scored[i], scored[i + 1])
            for i in range(len(scored) - 1)
            if float(scored[i]["score"]) > float(scored[i + 1]["score"])
        ]

    def bradley_terry_loss(
        self,
        chosen_logits: torch.Tensor,
        rejected_logits: torch.Tensor,
        label_smoothing: float | None = None,
    ) -> torch.Tensor:
        """Bradley-Terry loss over reward logits (mean over the batch).

        ``margin = chosen_logits - rejected_logits``; without smoothing the
        loss is ``-log_sigmoid(margin)`` (lower when the chosen sample's
        reward logit exceeds the rejected one's).  Pass ``label_smoothing``
        to override the engine default per call.
        """
        smoothing = self.label_smoothing if label_smoothing is None else label_smoothing
        if not 0.0 <= smoothing < 0.5:
            raise ValueError(f"label_smoothing must be in [0, 0.5), got {smoothing}")
        margin = chosen_logits - rejected_logits
        if smoothing > 0.0:
            loss = -(1.0 - smoothing) * F.logsigmoid(margin) - smoothing * F.logsigmoid(-margin)
        else:
            loss = -F.logsigmoid(margin)
        return loss.mean()

    def reward_from_critiques(self, feedback_list: list[dict[str, Any]]) -> dict[str, float]:
        """Map critique-scored records to a ``toon -> reward`` dict.

        Each record must carry a ``"toon"`` string and a numeric
        ``"score"`` (e.g. produced by pairing :class:`CritiqueFeedback`
        scores with their TOON strings); records without a toon are
        skipped.
        """
        rewards: dict[str, float] = {}
        for feedback in feedback_list:
            toon = feedback.get("toon")
            if not isinstance(toon, str) or not toon:
                continue
            try:
                score = float(feedback["score"])
            except (KeyError, TypeError, ValueError):
                continue
            rewards[toon] = max(0.0, min(1.0, score))
        return rewards
