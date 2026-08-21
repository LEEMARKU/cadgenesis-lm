"""tests/transformer/test_sparse_attention.py
=============================================
Unit tests for the sparse attention module (Pillar 1).
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.transformer.sparse_attention import (
    SPARSE_PATTERNS,
    SparseAttentionPattern,
    SparseSelfAttention,
    build_sparse_attention,
    pattern_complexity,
    sparse_attention_mask,
)


class TestSparseAttentionMask:
    def test_shape(self):
        mask = sparse_attention_mask("sliding_window", 16, window_size=8)
        assert mask.shape == (1, 1, 16, 16)

    def test_causal(self):
        mask = sparse_attention_mask("local", 8, window_size=4)
        assert mask[0, 0, 2, 5].item() == float("-inf")  # future masked

    def test_local_window_respects_window(self):
        mask = sparse_attention_mask("local", 10, window_size=3, causal=True)
        # query 7 may attend to keys 5,6,7 but not 4.
        assert mask[0, 0, 7, 7].item() == 0.0
        assert mask[0, 0, 7, 6].item() == 0.0
        assert mask[0, 0, 7, 4].item() == float("-inf")

    def test_global_tokens_attend_everything(self):
        mask = sparse_attention_mask("global", 10, num_global_tokens=2, causal=False)
        assert (mask[0, 0, 0] == 0.0).all()  # global query sees all
        assert (mask[0, 0, :, 0] == 0.0).all()  # all queries see global key
        assert mask[0, 0, 7, 3].item() == 0.0  # ordinary pair allowed

    def test_block_sparse_intra_block(self):
        mask = sparse_attention_mask(
            "block_sparse", 8, block_size=4, num_global_tokens=1, causal=False
        )
        assert mask[0, 0, 1, 3].item() == 0.0  # same block
        assert mask[0, 0, 5, 7].item() == 0.0  # same block
        assert mask[0, 0, 5, 2].item() == float("-inf")  # different blocks

    def test_mixed_is_union(self):
        mask = sparse_attention_mask("mixed", 10, window_size=3, num_global_tokens=2, causal=True)
        assert mask[0, 0, 9, 0].item() == 0.0  # global key
        assert mask[0, 0, 9, 7].item() == 0.0  # within band
        assert mask[0, 0, 9, 5].item() == float("-inf")

    def test_invalid_pattern_raises(self):
        with pytest.raises(ValueError):
            sparse_attention_mask("nystrom", 8)

    def test_all_patterns_enum(self):
        assert set(SPARSE_PATTERNS) == {
            "local",
            "global",
            "sliding_window",
            "block_sparse",
            "mixed",
        }
        assert SparseAttentionPattern.LOCAL.value == "local"


class TestSparseSelfAttention:
    @pytest.mark.parametrize("pattern", SPARSE_PATTERNS)
    def test_all_patterns_forward(self, pattern):
        attn = SparseSelfAttention(
            d_model=64,
            num_heads=4,
            pattern=pattern,
            window_size=8,
            num_global_tokens=2,
            block_size=4,
        )
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == (2, 16, 64)
        out.sum().backward()
        assert attn.q_proj.weight.grad is not None

    def test_attention_mask_combines(self):
        attn = SparseSelfAttention(d_model=64, num_heads=4, pattern="sliding_window", window_size=8)
        x = torch.randn(1, 8, 64)
        mask = torch.zeros(1, 1, 8, 8)
        out = attn(x, attn_mask=mask)
        assert out.shape == (1, 8, 64)
        assert torch.isfinite(out).all()

    def test_use_rope_false(self):
        attn = SparseSelfAttention(d_model=32, num_heads=2, pattern="local", window_size=4)
        x = torch.randn(1, 8, 32)
        out = attn(x, use_rope=False)
        assert out.shape == (1, 8, 32)

    def test_validation(self):
        with pytest.raises(ValueError):
            SparseSelfAttention(d_model=64, num_heads=3)
        with pytest.raises(ValueError):
            SparseSelfAttention(d_model=64, num_heads=4, pattern="nope")

    def test_sdpa_path(self):
        attn = SparseSelfAttention(
            d_model=32, num_heads=2, pattern="mixed", window_size=8, use_sdpa=True
        )
        x = torch.randn(1, 8, 32)
        assert attn(x).shape == (1, 8, 32)


class TestBuildSparseAttention:
    def test_factory(self):
        attn = build_sparse_attention(
            "block_sparse", d_model=64, num_heads=4, block_size=8, num_global_tokens=2
        )
        assert isinstance(attn, SparseSelfAttention)
        assert attn.pattern == "block_sparse"

    def test_complexity_text(self):
        text = pattern_complexity("sliding_window", 1024, window_size=128)
        assert "O(T·128)" in text and "dense" in text
