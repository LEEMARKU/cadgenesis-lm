"""tests/distillation/test_hard_labels.py"""

from __future__ import annotations

import torch

from cadgenesis.distillation.hard_labels import HardLabelExtractor


def make_logits() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(3)
    student = torch.randn(4, 6)
    teacher = torch.randn(4, 6)
    return student, teacher


def test_extract_argmax_labels_match_teacher():
    student, teacher = make_logits()
    batch = HardLabelExtractor().extract(student, teacher)
    assert torch.equal(batch.labels, teacher.argmax(dim=-1))
    assert batch.mask.dtype == torch.bool
    assert batch.mask.all()
    assert torch.allclose(batch.confidence, teacher.softmax(dim=-1).max(dim=-1).values)


def test_extract_rejects_shape_mismatch():
    with __import__("pytest").raises(ValueError):
        HardLabelExtractor().extract(torch.randn(2, 4), torch.randn(3, 4))


def test_mask_tokens_replaces_masked_positions():
    labels = torch.tensor([[3, 1, 0], [2, 5, 4]])
    mask = torch.tensor([[True, False, True], [True, True, False]])
    masked = HardLabelExtractor().mask_tokens(labels, mask, ignore_index=-100)
    assert torch.equal(masked, torch.tensor([[3, -100, 0], [2, 5, -100]]))


def test_mask_tokens_ignores_ignore_index_positions():
    student, teacher = make_logits()
    teacher[1, :] = -100.0
    batch = HardLabelExtractor().extract(student, teacher, ignore_index=-100)
    assert not batch.mask[1].any()


def test_min_confidence_masks_low_confidence_positions():
    teacher = torch.tensor([[5.0, 0.0, 0.0], [0.1, 0.1, 0.1]])
    batch = HardLabelExtractor(min_confidence=0.8).extract(torch.zeros_like(teacher), teacher)
    assert batch.mask[0].all()
    assert not batch.mask[1].any()
    assert bool(batch.confidence[1] <= 0.8)


def test_entropy_filter_flags_confident_predictions():
    probs = torch.tensor([[0.99, 0.005, 0.005], [0.34, 0.33, 0.33]])
    keep = HardLabelExtractor().entropy_filter(probs, threshold=0.5)
    assert keep[0].item() is True
    assert keep[1].item() is False


def test_entropy_filter_rejects_negative_threshold():
    with __import__("pytest").raises(ValueError):
        HardLabelExtractor().entropy_filter(torch.rand(2, 3), threshold=-0.1)


def test_rejects_invalid_min_confidence():
    with __import__("pytest").raises(ValueError):
        HardLabelExtractor(min_confidence=1.5)


def test_masked_labels_are_loss_ready():
    student, teacher = make_logits()
    # Row 1 is pre-ignored (logits all -100); everything else is kept, so
    # the masked target has both ignored and valid positions and the loss
    # is well-defined.  (Masking *every* position would make the
    # cross-entropy empty, which torch reports as nan.)
    teacher[1, :] = -100.0
    batch = HardLabelExtractor().extract(student, teacher, ignore_index=-100)
    masked = HardLabelExtractor().mask_tokens(batch.labels, batch.mask, ignore_index=-100)
    import torch.nn.functional as F

    loss = F.cross_entropy(student, masked, ignore_index=-100)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
