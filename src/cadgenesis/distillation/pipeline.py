"""cadgenesis.distillation.pipeline
================================
End-to-end distillation pipeline.

Orchestrates the existing distillation building blocks into a single
entry point -- it reuses (never reimplements) the classes from
``cadgenesis.distillation.distill_pipeline``:

1. :class:`AutomatedDatasetGenPipeline` -- teacher queries + quality
   filtering,
2. :class:`DistillationLossPipeline` -- soft (KL) + hard (CE) loss.

``run`` returns a :class:`DistillationRunReport` summarizing the filtered
dataset, hard-label coverage, the achieved pass rate and per-sample
distillation loss values (computed on small random student/teacher logits
over the dataset vocabulary so the report is self-contained and needs no
network).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from cadgenesis.distillation.distill_pipeline import (
    AutomatedDatasetGenPipeline,
    DistillationLossPipeline,
    QualityFilteringEngine,
    TeacherModelInterface,
)

__all__ = ["DistillationPipeline", "DistillationRunReport"]


@dataclass
class DistillationRunReport:
    """Summary of one :meth:`DistillationPipeline.run` execution.

    ``dataset`` holds the filtered ``{prompt, toon, objects}`` samples;
    ``hard_labels`` is the total number of teacher-argmax token positions
    across the dataset; ``pass_rate`` is the fraction of requested samples
    that survived quality filtering; ``distill_loss_values`` holds one
    combined (soft + hard) distillation loss per sample.
    """

    dataset: list[dict[str, Any]] = field(default_factory=list)
    hard_labels: int = 0
    pass_rate: float = 0.0
    distill_loss_values: list[float] = field(default_factory=list)


class DistillationPipeline:
    """End-to-end teacher -> filter -> label -> loss orchestration.

    Parameters
    ----------
    teacher:
        Any :class:`TeacherModelInterface` implementation (LLM-backed or
        rule-based fallback).
    quality_filter:
        Optional quality engine; a default :class:`QualityFilteringEngine`
        is created when omitted.
    temperature, alpha:
        Soft-target temperature and hard/soft loss mixing weight forwarded
        to :class:`DistillationLossPipeline`.
    """

    def __init__(
        self,
        teacher: TeacherModelInterface,
        quality_filter: QualityFilteringEngine | None = None,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ) -> None:
        self.teacher = teacher
        self.filter = quality_filter or QualityFilteringEngine()
        self.gen_pipeline = AutomatedDatasetGenPipeline(self.teacher, self.filter)
        self.loss_pipeline = DistillationLossPipeline(temperature=temperature, alpha=alpha)

    def generate_and_filter(self, num_samples: int) -> list[dict[str, Any]]:
        """Generate and quality-filter ``num_samples`` teacher samples.

        Delegates to :class:`AutomatedDatasetGenPipeline.generate_dataset`.
        """
        return self.gen_pipeline.generate_dataset(num_samples)

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Combined soft-KL + hard-CE distillation loss.

        Delegates to :class:`DistillationLossPipeline.compute_loss`.
        """
        return self.loss_pipeline.compute_loss(student_logits, teacher_logits, labels)

    def run(self, num_samples: int) -> DistillationRunReport:
        """Run the end-to-end pipeline over ``num_samples`` teacher queries."""
        dataset = self.generate_and_filter(num_samples)
        pass_rate = len(dataset) / num_samples if num_samples > 0 else 0.0
        return DistillationRunReport(
            dataset=dataset,
            hard_labels=self._count_hard_labels(dataset),
            pass_rate=round(pass_rate, 4),
            distill_loss_values=self._estimate_distill_losses(dataset),
        )

    def _estimate_distill_losses(
        self, dataset: list[dict[str, Any]], vocab_size: int = 16, seed: int = 0
    ) -> list[float]:
        """Per-sample distillation loss on proxy random logits.

        For each sample a label per object is derived from its feature
        (hashed into the vocabulary); student and teacher logits are random
        tensors of the same shape, so the loss exercises the full
        :class:`DistillationLossPipeline` math without needing a network or
        a real student model.
        """
        values: list[float] = []
        rng = torch.Generator().manual_seed(seed)
        for sample in dataset:
            objects = sample.get("objects") or []
            if not objects:
                continue
            labels = torch.tensor(
                [hash(str(obj.get("feature", ""))) % vocab_size for obj in objects],
                dtype=torch.long,
            )
            num_objects = labels.numel()
            student_logits = torch.randn(num_objects, vocab_size, generator=rng)
            teacher_logits = torch.randn(num_objects, vocab_size, generator=rng)
            loss = self.loss_pipeline.compute_loss(student_logits, teacher_logits, labels)
            values.append(float(loss.detach().item()))
        return values

    # ------------------------------------------------------------- report

    @staticmethod
    def _count_hard_labels(dataset: list[dict[str, Any]]) -> int:
        """Count teacher-argmax hard label positions over the dataset."""
        count = 0
        for sample in dataset:
            objects = sample.get("objects") or []
            if objects:
                count += sum(1 for obj in objects if str(obj.get("feature", "")).strip())
        return count
