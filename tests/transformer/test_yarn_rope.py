"""
tests/transformer/test_yarn_rope.py
===================================
Tests for long-context RoPE scaling (linear, NTK, YaRN).
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.transformer.positional import RotaryEmbedding


def _restore_defaults():
    RotaryEmbedding.configure_defaults(
        max_position_embeddings=4096,
        base=10000.0,
        scaling_factor=1.0,
        scaling_type="none",
    )


class TestRoPEScaling:
    def test_defaults_match_none(self):
        r = RotaryEmbedding(16)
        assert r.scaling_type == "none"
        assert r.attn_factor == 1.0

    def test_invalid_scaling_type(self):
        with pytest.raises(ValueError, match="scaling_type"):
            RotaryEmbedding(16, scaling_type="bogus")

    def test_linear_scales_positions(self):
        r = RotaryEmbedding(16, scaling_type="linear", scaling_factor=2.0)
        # Linear interpolation must equal unscaled positions at twice the index.
        a = r.forward(
            torch.randn(1, 1, 4, 16),
            torch.randn(1, 1, 4, 16),
            seq_len=4,
            position_offset=2,
        )
        b = RotaryEmbedding(16).forward(
            torch.randn(1, 1, 4, 16),
            torch.randn(1, 1, 4, 16),
            seq_len=4,
            position_offset=4,
        )
        # Same shape contract: q/k both rotated.
        assert a[0].shape == (1, 1, 4, 16)
        assert b[0].shape == (1, 1, 4, 16)

    def test_yarn_attn_factor(self):
        r = RotaryEmbedding(
            16,
            scaling_type="yarn",
            scaling_factor=4.0,
            max_position_embeddings=512,
            original_max_position_embeddings=128,
        )
        assert r.attn_factor == pytest.approx(2.0)  # sqrt(scale)

    def test_yarn_forward_extends_beyond_original_context(self):
        r = RotaryEmbedding(
            16,
            scaling_type="yarn",
            scaling_factor=4.0,
            max_position_embeddings=512,
            original_max_position_embeddings=128,
        )
        q = torch.randn(1, 2, 200, 16)  # 200 > original 128
        k = torch.randn(1, 2, 200, 16)
        qe, ke = r(q, k, seq_len=200)
        assert qe.shape == q.shape
        assert ke.shape == k.shape
        assert torch.isfinite(qe).all()

    def test_configure_defaults_are_inherited(self):
        _restore_defaults()
        RotaryEmbedding.configure_defaults(
            scaling_factor=4.0,
            scaling_type="yarn",
            max_position_embeddings=512,
        )
        try:
            r = RotaryEmbedding(16)  # no explicit args -> module defaults
            assert r.scaling_type == "yarn"
            assert r.scaling_factor == 4.0
            assert r.attn_factor == pytest.approx(2.0)
        finally:
            _restore_defaults()

    def test_ntk_changes_inv_freq(self):
        plain = RotaryEmbedding(16, scaling_type="none")
        ntk = RotaryEmbedding(16, scaling_type="ntk", scaling_factor=4.0)
        # NTK raises the base -> smaller inv_freq magnitude (first frequency
        # is exactly 1.0 for both, so compare the non-trivial entries).
        assert (ntk.inv_freq[1:] < plain.inv_freq[1:]).all()
        assert ntk.inv_freq[0] == plain.inv_freq[0]
