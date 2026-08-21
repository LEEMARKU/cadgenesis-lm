"""tests/transformer/test_multi_scale_attention.py
===================================================
Unit tests for the multi-scale attention module (Pillar 1).
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.transformer.multi_scale_attention import MultiScaleAttention


class TestMultiScaleAttention:
    def test_shape_and_grad(self):
        attn = MultiScaleAttention(d_model=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == (2, 16, 64)
        out.sum().backward()
        assert attn.q_proj.weight.grad is not None

    def test_head_split(self):
        attn = MultiScaleAttention(d_model=64, num_heads=4)
        report = attn.scale_report
        assert report["local"] + report["medium"] + report["global"] == 4
        assert report["global"] >= 1

    def test_default_split_matches(self):
        attn = MultiScaleAttention(d_model=64, num_heads=4)
        assert attn.local_heads == 2
        assert attn.medium_heads == 1
        assert attn.global_heads == 1

    def test_explicit_fractions(self):
        attn = MultiScaleAttention(
            d_model=64,
            num_heads=8,
            head_fractions=(0.25, 0.25, 0.5),
            local_window=16,
            medium_window=64,
        )
        assert attn.local_heads == 2
        assert attn.medium_heads == 2
        assert attn.global_heads == 4

    def test_validation(self):
        with pytest.raises(ValueError):
            MultiScaleAttention(d_model=64, num_heads=3)
        with pytest.raises(ValueError):
            MultiScaleAttention(d_model=64, num_heads=4, head_fractions=(1.0, 0.0, 0.0))
        with pytest.raises(ValueError):
            MultiScaleAttention(d_model=64, num_heads=4, local_window=64, medium_window=16)
        # Three scales require at least three heads.
        with pytest.raises(ValueError):
            MultiScaleAttention(d_model=32, num_heads=2)

    def test_causal_and_attn_mask(self):
        attn = MultiScaleAttention(
            d_model=32, num_heads=4, local_window=8, medium_window=16, causal=True
        )
        x = torch.randn(1, 12, 32)
        mask = torch.triu(torch.full((12, 12), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0)
        out = attn(x, attn_mask=mask)
        assert out.shape == (1, 12, 32)
        assert torch.isfinite(out).all()

    def test_sdpa_path(self):
        attn = MultiScaleAttention(d_model=32, num_heads=4, use_sdpa=True)
        x = torch.randn(1, 8, 32)
        assert attn(x).shape == (1, 8, 32)
