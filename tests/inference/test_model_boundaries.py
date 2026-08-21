"""tests/inference/test_model_boundaries.py
==========================================
Model robustness boundaries (mission section 6):

* vocabulary boundary (ids at cad_vocab_size-1 are legal; beyond -> error)
* sequence length boundary (block_size)
* NaN/Inf detection in logits and loss
* gradient stability on a small training step
"""

from __future__ import annotations

import math

import pytest
import torch

from cadgenesis.config import CADConfig
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer
from cadgenesis.transformer.losses import CADSequenceLoss as CADCausalLoss


@pytest.fixture(scope="module")
def model():
    cfg = CADConfig.mini()
    cfg.model.num_encoder_layers = 2
    cfg.model.num_decoder_layers = 2
    m = GeometryAwareTransformer(cfg)
    m.eval()
    return m


def test_vocabulary_upper_boundary(model):
    """The highest legal id must produce finite logits; beyond must error."""
    vocab = model.cad_vocab_size
    src = torch.zeros((1, 2), dtype=torch.long)
    tgt = torch.tensor([[vocab - 1, vocab - 1]], dtype=torch.long)
    typ = torch.zeros_like(tgt)
    with torch.no_grad():
        logits, _ = model(src, tgt, typ)
    assert logits.shape == (1, 2, vocab)
    assert torch.isfinite(logits).all(), "logits contain NaN/Inf at vocab boundary"

    with pytest.raises((IndexError, ValueError, RuntimeError)):
        model(src, torch.tensor([[vocab]], dtype=torch.long), typ)


def test_sequence_length_boundary(model):
    """block_size + 1 must fail clearly; block_size must work."""
    src = torch.zeros((1, 2), dtype=torch.long)
    seq_len = model.config.model.block_size
    tgt = torch.zeros((1, seq_len), dtype=torch.long)
    typ = torch.zeros_like(tgt)
    with torch.no_grad():
        logits, _ = model(src, tgt, typ)
    assert logits.shape[1] == seq_len
    assert torch.isfinite(logits).all()

    with pytest.raises((IndexError, ValueError, RuntimeError)):
        model(src, torch.zeros((1, seq_len + 1), dtype=torch.long), typ)


def test_loss_finite_on_clean_batch(model):
    """Training loss on a clean batch must be finite and positive."""
    src = torch.zeros((2, 4), dtype=torch.long)
    tgt_in = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    typ = torch.zeros_like(tgt_in)
    target = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]], dtype=torch.long)
    mask = torch.ones_like(target, dtype=torch.bool)

    logits, _ = model(src, tgt_in, typ)
    loss = CADCausalLoss()(logits, target, mask)[0]
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    assert loss > 0.0, f"loss must be positive: {loss}"


def test_loss_handles_nan_inf_targets(model):
    """NaN/Inf in logits must be detected by the loss (never silently kept)."""
    vocab = model.cad_vocab_size
    logits = torch.full((1, 2, vocab), float("nan"))
    target = torch.tensor([[1, 2]], dtype=torch.long)
    mask = torch.ones_like(target, dtype=torch.bool)
    loss = CADCausalLoss()(logits, target, mask)[0]
    assert not torch.isfinite(loss), "NaN logits must produce NaN loss (visible, not silent)"


def test_gradient_step_is_stable(model):
    """One optimizer step must produce finite gradients and reduce loss."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    src = torch.zeros((2, 4), dtype=torch.long)
    tgt_in = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    typ = torch.zeros_like(tgt_in)
    target = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]], dtype=torch.long)
    mask = torch.ones_like(target, dtype=torch.bool)

    logits, _ = model(src, tgt_in, typ)
    loss = CADCausalLoss()(logits, target, mask)[0]
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    for g in grads:
        assert torch.isfinite(g).all(), "non-finite gradient found"
    optimizer.step()
    optimizer.zero_grad()

    logits2, _ = model(src, tgt_in, typ)
    loss2 = CADCausalLoss()(logits2, target, mask)[0]
    assert loss2 < loss, f"loss did not decrease: {loss.item():.4f} -> {loss2.item():.4f}"
    assert math.isfinite(loss2.item())
