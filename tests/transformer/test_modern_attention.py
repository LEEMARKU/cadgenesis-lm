"""tests/transformer/test_modern_attention.py
============================================
Unit tests for modern attention mechanisms (GQA & MLA).
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.transformer.modern_attention import (
    GroupedQueryAttention,
    MultiHeadLatentAttention,
)


class TestGroupedQueryAttention:
    def test_shapes(self):
        attn = GroupedQueryAttention(d_model=64, num_heads=4, num_kv_heads=1)
        x = torch.randn(2, 8, 64)
        out = attn(x)
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()

    def test_defaults_to_single_kv_head(self):
        attn = GroupedQueryAttention(d_model=64, num_heads=4)
        assert attn.num_kv_heads == 1
        assert attn.k_proj.out_features == 16
        assert attn.v_proj.out_features == 16
        assert attn.q_proj.out_features == 64

    def test_dropout_training_vs_eval(self):
        attn = GroupedQueryAttention(d_model=64, num_heads=4, num_kv_heads=2, dropout=0.5)
        x = torch.randn(2, 8, 64)
        attn.train()
        o1 = attn(x)
        o2 = attn(x)
        assert not torch.allclose(o1, o2)
        attn.eval()
        o3 = attn(x)
        o4 = attn(x)
        assert torch.allclose(o3, o4)

    def test_attn_mask(self):
        attn = GroupedQueryAttention(d_model=64, num_heads=4, num_kv_heads=2)
        x = torch.randn(2, 8, 64)
        mask = torch.zeros(2, 1, 8, 8)
        out = attn(x, attn_mask=mask)
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()

    def test_use_rope_false(self):
        attn = GroupedQueryAttention(d_model=64, num_heads=4, num_kv_heads=1)
        x = torch.randn(2, 8, 64)
        out = attn(x, use_rope=False)
        assert out.shape == (2, 8, 64)

    def test_invalid_num_kv_heads_raises(self):
        with pytest.raises(ValueError):
            GroupedQueryAttention(d_model=64, num_heads=4, num_kv_heads=0)
        with pytest.raises(ValueError):
            GroupedQueryAttention(d_model=64, num_heads=4, num_kv_heads=5)
        with pytest.raises(ValueError):
            GroupedQueryAttention(d_model=64, num_heads=4, num_kv_heads=-1)

    def test_num_heads_divisible_by_num_kv_heads_enforced(self):
        with pytest.raises(ValueError):
            GroupedQueryAttention(d_model=64, num_heads=4, num_kv_heads=3)

    def test_d_model_divisible_by_num_heads_enforced(self):
        with pytest.raises(ValueError):
            GroupedQueryAttention(d_model=64, num_heads=3)

    def test_matches_manual_reference(self):
        attn = GroupedQueryAttention(d_model=64, num_heads=4, num_kv_heads=4)
        attn.eval()
        x = torch.randn(2, 8, 64)

        q = attn.q_proj(x).view(2, 8, 4, 16).transpose(1, 2)
        k = attn.k_proj(x).view(2, 8, 4, 16).transpose(1, 2)
        v = attn.v_proj(x).view(2, 8, 4, 16).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (16**0.5)
        probs = torch.softmax(scores, dim=-1)
        manual = torch.matmul(probs, v).transpose(1, 2).contiguous().view(2, 8, 64)
        manual = attn.out_proj(manual)

        out = attn(x, attn_mask=torch.zeros(2, 1, 8, 8), use_rope=False)
        assert torch.allclose(out, manual, atol=1e-4, rtol=1e-4)


class TestMultiHeadLatentAttention:
    def test_shapes(self):
        attn = MultiHeadLatentAttention(
            d_model=128, num_heads=4, kv_lora_rank=16, qk_rope_head_dim=8
        )
        x = torch.randn(2, 8, 128)
        out = attn(x)
        assert out.shape == (2, 8, 128)
        assert torch.isfinite(out).all()

    def test_attn_mask(self):
        attn = MultiHeadLatentAttention(
            d_model=128, num_heads=4, kv_lora_rank=16, qk_rope_head_dim=8
        )
        x = torch.randn(2, 8, 128)
        mask = torch.zeros(2, 1, 8, 8)
        out = attn(x, attn_mask=mask)
        assert out.shape == (2, 8, 128)
        assert torch.isfinite(out).all()

    def test_use_rope_false(self):
        attn = MultiHeadLatentAttention(
            d_model=128, num_heads=4, kv_lora_rank=16, qk_rope_head_dim=8
        )
        x = torch.randn(2, 8, 128)
        out = attn(x, use_rope=False)
        assert out.shape == (2, 8, 128)
        assert torch.isfinite(out).all()

    def test_kv_cache_savings_ratio(self):
        attn = MultiHeadLatentAttention(
            d_model=128, num_heads=4, kv_lora_rank=16, qk_rope_head_dim=8
        )
        assert attn.kv_cache_savings_ratio() < 1
        assert attn.kv_cache_savings_ratio() == 1 - 16 / (2 * 32)

    def test_kv_latent_exposed(self):
        attn = MultiHeadLatentAttention(
            d_model=128, num_heads=4, kv_lora_rank=16, qk_rope_head_dim=8
        )
        x = torch.randn(2, 8, 128)
        attn(x)
        assert attn.last_kv_latent is not None
        assert attn.last_kv_latent.shape == (2, 8, 16)
        assert not attn.last_kv_latent.requires_grad

    def test_rope_dim_validated(self):
        with pytest.raises(ValueError):
            MultiHeadLatentAttention(d_model=128, num_heads=4, qk_rope_head_dim=40)
        with pytest.raises(ValueError):
            MultiHeadLatentAttention(d_model=128, num_heads=4, qk_rope_head_dim=0)

    def test_up_projs_bias_free(self):
        attn = MultiHeadLatentAttention(
            d_model=128, num_heads=4, kv_lora_rank=16, qk_rope_head_dim=8
        )
        assert attn.w_uk.bias is None
        assert attn.w_uv.bias is None


class TestGradients:
    def test_gqa_gradients_flow(self):
        attn = GroupedQueryAttention(d_model=64, num_heads=4, num_kv_heads=2)
        x = torch.randn(2, 8, 64)
        out = attn(x)
        out.sum().backward()
        for name, p in attn.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"

    def test_mla_gradients_flow(self):
        attn = MultiHeadLatentAttention(
            d_model=128, num_heads=4, kv_lora_rank=16, qk_rope_head_dim=8
        )
        x = torch.randn(2, 8, 128)
        out = attn(x)
        out.sum().backward()
        for name, p in attn.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"
