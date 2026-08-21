"""tests/distillation/test_soft_labels.py"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cadgenesis.distillation.soft_labels import SoftLabelGenerator


def test_soft_targets_matches_plain_softmax_at_unit_temperature():
    logits = torch.randn(3, 5)
    probs = SoftLabelGenerator.soft_targets(logits, temperature=1.0)
    assert torch.allclose(probs, F.softmax(logits, dim=-1), atol=1e-6)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_soft_targets_temperature_scaling():
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    cold = SoftLabelGenerator.soft_targets(logits, temperature=0.5)
    hot = SoftLabelGenerator.soft_targets(logits, temperature=5.0)
    manual_cold = F.softmax(logits / 0.5, dim=-1)
    assert torch.allclose(cold, manual_cold, atol=1e-6)
    assert cold[0, 0] > hot[0, 0]
    assert hot[0, 0] > 1.0 / 3.0


def test_soft_targets_smoothing_is_uniform_mixture():
    logits = torch.randn(2, 4)
    smoothing = 0.3
    probs = SoftLabelGenerator.soft_targets(logits, temperature=2.0, smoothing=smoothing)
    raw = F.softmax(logits / 2.0, dim=-1)
    manual = (1.0 - smoothing) * raw + smoothing / 4.0
    assert torch.allclose(probs, manual, atol=1e-6)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_kl_loss_matches_manual_math():
    student = torch.randn(4, 8)
    teacher = torch.randn(4, 8)
    temperature = 2.0
    loss = SoftLabelGenerator.kl_loss(student, teacher, temperature)
    manual = (
        F.kl_div(
            F.log_softmax(student / temperature, dim=-1),
            F.softmax(teacher / temperature, dim=-1),
            reduction="batchmean",
        )
        * temperature**2
    )
    assert torch.allclose(loss, manual, atol=1e-6)
    assert loss.ndim == 0


def test_kl_loss_is_zero_for_identical_logits():
    logits = torch.randn(3, 6)
    loss = SoftLabelGenerator.kl_loss(logits, logits, temperature=2.0)
    assert loss.item() == 0.0


def test_kl_loss_rejects_shape_mismatch():
    with __import__("pytest").raises(ValueError):
        SoftLabelGenerator.kl_loss(torch.randn(3, 5), torch.randn(3, 6))


def test_soften_labels_smoothed_one_hot():
    labels = torch.tensor([0, 2, 1])
    num_classes = 4
    smoothing = 0.2
    soft = SoftLabelGenerator.soften_labels(labels, num_classes, smoothing)
    assert soft.shape == (3, 4)
    assert torch.allclose(soft.sum(dim=-1), torch.ones(3), atol=1e-6)
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    manual = (1.0 - smoothing) * one_hot + smoothing / num_classes
    assert torch.allclose(soft, manual, atol=1e-6)
    assert soft[0, 0] == (1.0 - smoothing) + smoothing / num_classes


def test_soften_labels_zero_smoothing_is_one_hot():
    labels = torch.tensor([1, 0])
    soft = SoftLabelGenerator.soften_labels(labels, 3, 0.0)
    assert torch.allclose(soft, F.one_hot(labels, num_classes=3).float())


def test_rejects_invalid_smoothing():
    import pytest

    with pytest.raises(ValueError):
        SoftLabelGenerator.soft_targets(torch.randn(2, 3), smoothing=1.5)
    with pytest.raises(ValueError):
        SoftLabelGenerator.soften_labels(torch.zeros(2, dtype=torch.long), 3, -0.1)
    with pytest.raises(ValueError):
        SoftLabelGenerator.soften_labels(torch.zeros(2, dtype=torch.long), 0, 0.1)


def test_loss_is_differentiable():
    student = torch.randn(2, 4, requires_grad=True)
    teacher = torch.randn(2, 4)
    loss = SoftLabelGenerator.kl_loss(student, teacher)
    loss.backward()
    assert student.grad is not None
