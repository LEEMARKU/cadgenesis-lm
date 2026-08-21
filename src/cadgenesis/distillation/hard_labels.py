"""cadgenesis.distillation.hard_labels
=====================================
Hard-label distillation.

Hard-label distillation trains the student with standard cross-entropy on
the teacher's *argmax* predictions::

    y_teacher = argmax_v z_teacher          # hard label
    L_hard     = cross_entropy(z_student, y_teacher)

Because teacher argmaxes can be noisy, the extractor also provides
confidence filtering (entropy or top-probability thresholds) so that
low-confidence positions can be masked out with ``ignore_index``.

Contract notes
--------------
* ``HardLabelExtractor.extract`` validates that ``student_logits`` and
  ``teacher_logits`` share the same shape, then derives hard labels from
  the teacher, a boolean ``mask`` (True = keep) based on ``min_confidence``,
  and per-position ``confidence``.
* ``mask_tokens`` replaces masked positions with ``ignore_index`` so the
  result can be fed straight into ``torch.nn.functional.cross_entropy``.
* ``entropy_filter`` returns True where the (clamped) Shannon entropy of
  the probability vector is below ``threshold`` -- low entropy means a
  confident, unambiguous teacher prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

__all__ = ["HardLabelBatch", "HardLabelExtractor"]


@dataclass
class HardLabelBatch:
    """Hard labels plus the mask/confidence metadata needed for training.

    ``labels`` are the teacher argmax tokens; ``mask`` is True where the
    position should participate in the loss; ``confidence`` is the teacher's
    top-1 softmax probability at each position.
    """

    labels: torch.Tensor
    mask: torch.Tensor
    confidence: torch.Tensor


class HardLabelExtractor:
    """Extracts filtered hard labels from teacher logits.

    Parameters
    ----------
    min_confidence:
        Positions whose teacher top-1 probability is below this value are
        masked out (default ``0.0`` keeps everything).
    """

    def __init__(self, min_confidence: float = 0.0) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")
        self.min_confidence = min_confidence

    def extract(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ignore_index: int = -100,
    ) -> HardLabelBatch:
        """Build a :class:`HardLabelBatch` from teacher logits.

        ``student_logits`` is only used for shape validation (the student
        and teacher must operate on the same vocabulary).  ``labels`` are
        the teacher argmaxes; positions already set to ``ignore_index`` are
        always masked out.
        """
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                "student_logits and teacher_logits must share the same shape, "
                f"got {tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
            )
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        labels = teacher_logits.argmax(dim=-1)
        confidence = teacher_probs.max(dim=-1).values
        # Positions whose teacher argmax value is exactly ``ignore_index``
        # (e.g. an entire row pre-filled with -100) are already ignored:
        # their label is replaced with ``ignore_index`` and they never enter
        # the loss.
        argmax_values = teacher_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        ignored = argmax_values == ignore_index
        labels = torch.where(ignored, torch.full_like(labels, ignore_index), labels)
        mask = (confidence >= self.min_confidence) & ~ignored
        return HardLabelBatch(labels=labels, mask=mask, confidence=confidence)

    def mask_tokens(
        self, labels: torch.Tensor, mask: torch.Tensor, ignore_index: int = -100
    ) -> torch.Tensor:
        """Replace masked positions with ``ignore_index``.

        ``mask`` must be a boolean tensor broadcastable to ``labels``
        (typically the ``mask`` field of a :class:`HardLabelBatch`).
        """
        if mask.shape != labels.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} does not match labels shape {tuple(labels.shape)}"
            )
        return torch.where(mask, labels, torch.full_like(labels, ignore_index))

    def entropy_filter(self, probs: torch.Tensor, threshold: float) -> torch.Tensor:
        """Boolean mask of positions whose entropy is below ``threshold``.

        Entropy is computed over the last dimension with probabilities
        clamped at ``1e-12`` to avoid ``log(0)``.  Returns a boolean tensor
        of shape ``probs.shape[:-1]``, True = confident (keep).
        """
        if threshold < 0.0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        safe_probs = probs.clamp_min(1e-12)
        entropy = -(safe_probs * safe_probs.log()).sum(dim=-1)
        return entropy <= threshold
