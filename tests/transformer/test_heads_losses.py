"""tests/transformer/test_heads_losses.py
========================================
Unit tests for cadgenesis.transformer.heads and .losses.
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.transformer.heads import (
    ConfidenceHead,
    LMHead,
    OutputHeads,
)
from cadgenesis.transformer.losses import (
    CADSequenceLoss,
    ConfidenceLoss,
    MaskedCrossEntropyLoss,
)


class TestLMHead:
    def test_shape(self):
        head = LMHead(d_model=32, vocab_size=100)
        out = head(torch.randn(2, 16, 32))
        assert out.shape == (2, 16, 100)

    def test_weight_tie(self):
        import torch.nn as nn

        emb = nn.Embedding(100, 32)
        head = LMHead(d_model=32, vocab_size=100, tie_weights=emb)
        assert head.is_tied
        assert head.out_proj.weight is emb.weight

    def test_tie_mismatch(self):
        import torch.nn as nn

        emb = nn.Embedding(50, 32)
        with pytest.raises(ValueError):
            LMHead(d_model=32, vocab_size=100, tie_weights=emb)

    def test_validation(self):
        with pytest.raises(ValueError):
            LMHead(d_model=0, vocab_size=10)


class TestConfidenceHead:
    def test_shape(self):
        head = ConfidenceHead(d_model=32)
        assert head(torch.randn(2, 16, 32)).shape == (2, 16, 1)


class TestOutputHeads:
    def test_shapes(self):
        heads = OutputHeads(d_model=32, vocab_size=100)
        logits, conf = heads(torch.randn(2, 16, 32))
        assert logits.shape == (2, 16, 100)
        assert conf.shape == (2, 16, 1)


class TestMaskedCrossEntropyLoss:
    def test_basic(self):
        loss = MaskedCrossEntropyLoss()
        logits = torch.randn(2, 4, 10)
        targets = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]])
        value = loss(logits, targets)
        assert value.shape == ()
        assert value.item() > 0

    def test_padding_masked(self):
        loss = MaskedCrossEntropyLoss()
        logits = torch.randn(2, 4, 10)
        targets = torch.zeros(2, 4, dtype=torch.long)
        value = loss(logits, targets)
        assert value.item() == 0.0

    def test_explicit_mask(self):
        loss = MaskedCrossEntropyLoss()
        logits = torch.randn(2, 4, 10)
        targets = torch.randint(0, 10, (2, 4))
        mask = torch.tensor([[True, True, True, False], [True, True, True, False]])
        assert loss(logits, targets, mask=mask).shape == ()


class TestConfidenceLoss:
    def test_basic(self):
        loss = ConfidenceLoss()
        logits = torch.randn(2, 4, 1)
        targets = torch.randint(0, 2, (2, 4)).float()
        assert loss(logits, targets).item() >= 0

    def test_empty_masked(self):
        loss = ConfidenceLoss()
        logits = torch.randn(2, 4, 1)
        targets = torch.rand(2, 4)
        mask = torch.zeros(2, 4, dtype=torch.bool)
        assert loss(logits, targets, mask=mask).item() == 0.0

    def test_invalid_reduction(self):
        with pytest.raises(ValueError):
            ConfidenceLoss(reduction="nope")


class TestCADSequenceLoss:
    def test_total_breakdown(self):
        loss = CADSequenceLoss()
        logits = torch.randn(2, 4, 10)
        targets = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]])
        conf_logits = torch.randn(2, 4, 1)
        target_conf = torch.randint(0, 2, (2, 4)).float()
        total, breakdown = loss(
            logits, targets, confidence_logits=conf_logits, target_confidence=target_conf
        )
        assert set(breakdown) >= {"ce", "confidence", "total"}
        assert total.item() == pytest.approx(breakdown["total"], rel=1e-5)

    def test_with_aux(self):
        loss = CADSequenceLoss(moe_aux_scale=0.5)
        logits = torch.randn(2, 4, 10)
        targets = torch.randint(0, 10, (2, 4))
        aux = torch.tensor(1.0)
        total, breakdown = loss(logits, targets, aux_loss=aux)
        assert breakdown["moe_aux"] == pytest.approx(0.5)
        assert total.item() == pytest.approx(breakdown["ce"] + 0.5)

    def test_negative_weight(self):
        with pytest.raises(ValueError):
            CADSequenceLoss(confidence_weight=-0.1)
