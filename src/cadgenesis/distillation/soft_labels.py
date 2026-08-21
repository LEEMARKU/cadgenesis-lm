"""cadgenesis.distillation.soft_labels
=====================================
Soft-label distillation.

Soft-label (Hinton et al., 2015) distillation trains the student against
the teacher's *soft probabilities* rather than one-hot hard labels.  The
teacher distribution is sharpened/softened with a temperature ``T``::

    p_teacher = softmax(z_teacher / T)         # teacher soft target
    p_student = log_softmax(z_student / T)     # student (log-space)
    L_soft     = T^2 * KL(p_student || p_teacher)

The ``T^2`` factor restores the scale of the logits so the gradient
magnitude is independent of the temperature; this is exactly the math used
by :class:`MultiTeacherDistillationEngine` (see
``cadgenesis.distillation.distillation_engine``).

Contract notes
--------------
* ``SoftLabelGenerator.soft_targets`` returns probabilities (rows sum to
  1).  When ``smoothing > 0`` the target becomes a mixture with the uniform
  distribution: ``(1 - s) * p + s * U``, which acts as a regularizer
  against over-confident teacher targets.
* ``SoftLabelGenerator.kl_loss`` returns a scalar tensor with
  ``reduction="batchmean"`` (true mean over the flattened batch).
* ``SoftLabelGenerator.soften_labels`` turns integer class ids into
  smoothed one-hot vectors for the *labels* branch of distillation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["SoftLabelGenerator"]


class SoftLabelGenerator:
    """Stateless helpers for soft-label (KD) targets and losses.

    All methods are static and operate purely on tensors; the class acts as
    a namespace documenting the exact math shared with
    :class:`~cadgenesis.distillation.distillation_engine.MultiTeacherDistillationEngine`.
    """

    @staticmethod
    def soft_targets(
        logits: torch.Tensor, temperature: float = 1.0, smoothing: float = 0.0
    ) -> torch.Tensor:
        """Temperature-scaled teacher probabilities with optional smoothing.

        ``probs = softmax(logits / temperature)``; when ``smoothing > 0``
        the result is mixed with the uniform distribution over the last
        dimension: ``(1 - smoothing) * probs + smoothing / V``.  Rows always
        sum to 1.  ``smoothing`` must lie in ``[0, 1]``.
        """
        if not 0.0 <= smoothing <= 1.0:
            raise ValueError(f"smoothing must be in [0, 1], got {smoothing}")
        probs = F.softmax(logits / temperature, dim=-1)
        if smoothing > 0.0:
            num_classes = logits.shape[-1]
            uniform = torch.full_like(probs, 1.0 / num_classes)
            probs = (1.0 - smoothing) * probs + smoothing * uniform
        return probs

    @staticmethod
    def kl_loss(
        student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0
    ) -> torch.Tensor:
        """Temperature-scaled KL divergence between student and teacher.

        Matches :class:`MultiTeacherDistillationEngine` exactly::

            KL(log_softmax(student / T) || softmax(teacher / T)) * T^2

        with ``reduction="batchmean"``.  ``student_logits`` and
        ``teacher_logits`` must share the same shape; the last dimension is
        the vocabulary.
        """
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                "student_logits and teacher_logits must share the same shape, "
                f"got {tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
            )
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)
        soft_teacher = F.log_softmax(teacher_logits / temperature, dim=-1)
        return F.kl_div(soft_student, soft_teacher, log_target=True, reduction="batchmean") * (
            temperature**2
        )

    @staticmethod
    def soften_labels(labels: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
        """Convert integer labels into smoothed one-hot vectors.

        Returns a float tensor of shape ``labels.shape + (num_classes,)``
        where the on-hot entry has weight ``1 - smoothing`` and every class
        receives ``smoothing / num_classes``.  ``smoothing`` must lie in
        ``[0, 1]``.
        """
        if not 0.0 <= smoothing <= 1.0:
            raise ValueError(f"smoothing must be in [0, 1], got {smoothing}")
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        one_hot = F.one_hot(labels.long(), num_classes=num_classes).float()
        uniform = torch.full(
            (num_classes,), smoothing / num_classes, dtype=one_hot.dtype, device=one_hot.device
        )
        return (1.0 - smoothing) * one_hot + uniform
