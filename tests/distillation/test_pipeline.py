"""tests/distillation/test_pipeline.py"""

from __future__ import annotations

import torch

from cadgenesis.distillation.distill_pipeline import (
    AutomatedDatasetGenPipeline,
    DistillationLossPipeline,
    TeacherModelInterface,
)
from cadgenesis.distillation.pipeline import DistillationPipeline


def test_run_end_to_end_no_network():
    pipeline = DistillationPipeline(teacher=TeacherModelInterface())
    report = pipeline.run(num_samples=5)
    assert len(report.dataset) >= 1
    assert 0.0 <= report.pass_rate <= 1.0
    assert report.hard_labels == sum(len(sample["objects"]) for sample in report.dataset)
    assert len(report.distill_loss_values) == len(report.dataset)
    assert all(torch.isfinite(torch.tensor(value)) for value in report.distill_loss_values)


def test_generate_and_filter_delegates_to_automated_pipeline():
    pipeline = DistillationPipeline(teacher=TeacherModelInterface())
    dataset = pipeline.generate_and_filter(num_samples=5)
    assert isinstance(dataset, list)
    assert len(dataset) >= 1
    assert all(set(sample) == {"prompt", "toon", "objects"} for sample in dataset)
    delegated = AutomatedDatasetGenPipeline(pipeline.teacher, pipeline.filter).generate_dataset(5)
    assert len(delegated) >= 1


def test_compute_loss_delegates_to_loss_pipeline():
    pipeline = DistillationPipeline(teacher=TeacherModelInterface(), temperature=2.0, alpha=0.5)
    torch.manual_seed(0)
    student = torch.randn(4, 6)
    teacher = torch.randn(4, 6)
    labels = torch.randint(0, 6, (4,))
    ours = pipeline.compute_loss(student, teacher, labels)
    reference = DistillationLossPipeline(temperature=2.0, alpha=0.5).compute_loss(
        student, teacher, labels
    )
    assert torch.allclose(ours, reference, atol=1e-6)
    assert ours.ndim == 0


def test_run_honors_custom_temperature_and_alpha():
    pipeline = DistillationPipeline(teacher=TeacherModelInterface(), temperature=4.0, alpha=0.9)
    report = pipeline.run(num_samples=3)
    assert report.distill_loss_values
    torch.manual_seed(0)
    student = torch.randn(3, 5)
    teacher = torch.randn(3, 5)
    labels = torch.randint(0, 5, (3,))
    reference = DistillationLossPipeline(temperature=4.0, alpha=0.9).compute_loss(
        student, teacher, labels
    )
    assert torch.allclose(pipeline.compute_loss(student, teacher, labels), reference)


def test_all_ignored_labels_yield_finite_loss():
    """PyTorch's cross_entropy returns NaN when every label is ignored; the
    distillation engine must guard against it (v6.2 fix, caught by
    test_run_end_to_end_no_network flakiness from PYTHONHASHSEED)."""
    engine = DistillationLossPipeline(temperature=2.0, alpha=0.5)
    student = torch.randn(1, 16)
    teacher = torch.randn(1, 16)
    loss = engine.compute_loss(student, teacher, torch.tensor([0]))
    assert torch.isfinite(loss).item()
    assert float(loss) > 0.0  # the soft (KL) term still contributes


def test_run_zero_samples_is_safe():
    report = DistillationPipeline(teacher=TeacherModelInterface()).run(num_samples=0)
    assert report.dataset == []
    assert report.pass_rate == 0.0
    assert report.hard_labels == 0
    assert report.distill_loss_values == []
