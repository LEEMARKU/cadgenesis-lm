"""
tests/quantization/test_bitnet.py
=================================
Tests for BitNet b1.58 (ternary weights + int8 activations, straight-through).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cadgenesis.quantization.bitnet import BitLinear, apply_bitnet


class TestBitLinear:
    def test_ternarize_values(self):
        w = torch.tensor([[0.9, -0.4, 0.0, 1.1]])
        scale = w.abs().mean().item()
        t = BitLinear._ternarize(w)
        vals = torch.unique(t.round(decimals=3))
        assert sorted(vals.tolist()) == sorted({-scale, 0.0, scale})

    def test_forward_shape_and_quantized_act(self):
        torch.manual_seed(0)
        bit = BitLinear(8, 16)
        x = torch.randn(4, 8)
        out = bit(x)
        assert out.shape == (4, 16)
        assert torch.isfinite(out).all()

    def test_straight_through_gradients_flow_to_fp_weights(self):
        torch.manual_seed(0)
        bit = BitLinear(8, 16)
        x = torch.randn(4, 8, requires_grad=True)
        out = bit(x).sum()
        out.backward()
        assert x.grad is not None
        assert bit.weight.grad is not None
        assert bit.weight.grad.abs().sum() > 0


class TestApplyBitnet:
    def test_swaps_linears_and_skips_tied(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
        # Tie the last linear to the first (weight sharing via out_proj->emb).
        model[1].weight = model[0].weight
        count = apply_bitnet(model)
        # Both linears share the same weight tensor -> neither may be quantised
        # (a quantised + fp32 path over one shared weight would break grad flow).
        assert count == 0
        assert not isinstance(model[0], BitLinear)
        assert not isinstance(model[1], BitLinear)

    def test_exclude_pattern(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
        count = apply_bitnet(model, exclude=("1",))
        assert count == 1
        assert isinstance(model[0], BitLinear)
        assert not isinstance(model[1], BitLinear)

    def test_preserves_weights(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(8, 8))
        original = model[0].weight.clone()
        apply_bitnet(model)
        assert torch.equal(model[0].weight, original)
