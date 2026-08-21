"""tests/transformer/test_rope_scaling.py
=========================================
Unit tests for the configurable long-context RoPE scaling added in Pillar 1.
The default ``scaling_type="none"`` must remain byte-for-byte compatible with
the legacy behaviour.
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.transformer.positional import RotaryEmbedding


class TestRoPEScaling:
    def test_default_unchanged(self):
        a = RotaryEmbedding(dim=32, max_position_embeddings=64)
        b = RotaryEmbedding(
            dim=32, max_position_embeddings=64, scaling_factor=1.0, scaling_type="none"
        )
        q = torch.randn(2, 4, 16, 32)
        k = torch.randn(2, 4, 16, 32)
        qa, ka = a(q, k)
        qb, kb = b(q, k)
        assert torch.allclose(qa, qb)
        assert torch.allclose(ka, kb)

    def test_linear_scaling_differs(self):
        plain = RotaryEmbedding(dim=32, max_position_embeddings=64)
        scaled = RotaryEmbedding(
            dim=32, max_position_embeddings=64, scaling_factor=4.0, scaling_type="linear"
        )
        q = torch.randn(1, 2, 32, 32)
        k = torch.randn(1, 2, 32, 32)
        _, ka = scaled(q, k)
        _, kp = plain(q, k)
        assert not torch.allclose(ka, kp)

    def test_linear_scaling_extends_context(self):
        # With factor 4, positions are compressed 4x so long sequences stay
        # within the cached table's rotation range.
        scaled = RotaryEmbedding(
            dim=32, max_position_embeddings=64, scaling_factor=4.0, scaling_type="linear"
        )
        q = torch.randn(1, 2, 256, 32)  # 4x the trained context
        k = torch.randn(1, 2, 256, 32)
        q_out, _ = scaled(q, k)
        assert q_out.shape == (1, 2, 256, 32)
        assert torch.isfinite(q_out).all()

    def test_ntk_scaling(self):
        scaled = RotaryEmbedding(dim=32, scaling_factor=2.0, scaling_type="ntk")
        plain = RotaryEmbedding(dim=32)
        # NTK raises the base frequency -> different rotations.
        q = torch.randn(1, 1, 8, 32)
        k = torch.randn(1, 1, 8, 32)
        _, ks = scaled(q, k)
        _, kp = plain(q, k)
        assert not torch.allclose(ks, kp)

    def test_ntk_base_raised(self):
        scaled = RotaryEmbedding(dim=32, scaling_factor=2.0, scaling_type="ntk")
        plain = RotaryEmbedding(dim=32)
        effective_base = 10000.0 * (2.0 ** (32 / 30))
        # inv_freq[i] = 1 / base^((2i) / dim); index 1 -> exponent 2/32.
        inv1 = float(scaled.inv_freq[1].item())
        assert inv1 == pytest.approx(1.0 / (effective_base ** (2 / 32)))
        assert inv1 < float(plain.inv_freq[1].item())  # raised base -> smaller freq

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            RotaryEmbedding(dim=32, scaling_factor=0.0)
        with pytest.raises(ValueError):
            RotaryEmbedding(dim=32, scaling_type="bogus")

    def test_yarn_now_supported(self):
        rope = RotaryEmbedding(dim=32, scaling_factor=4.0, scaling_type="yarn")
        assert rope.attn_factor == pytest.approx(2.0)

    def test_forward_longer_than_cache_dynamic(self):
        rope = RotaryEmbedding(dim=32, max_position_embeddings=32)
        q = torch.randn(1, 1, 128, 32)
        k = torch.randn(1, 1, 128, 32)
        q_out, _ = rope(q, k)
        assert q_out.shape == (1, 1, 128, 32)
        assert torch.isfinite(q_out).all()
