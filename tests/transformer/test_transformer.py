"""
tests/model/test_transformer.py
================================
Unit tests for GeometryAwareTransformer & attention heads.
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.config import CADConfig
from cadgenesis.transformer.attention import (
    AgentAttention,
    ConstraintAttention,
    GeometryAttention,
    MemoryAttention,
    MultiHeadAttentionMixture,
    SelfAttention,
    UncertaintyAttention,
)
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer
from cadgenesis.transformer.positional import (
    ALiBiBias,
    RotaryEmbedding,
    SinusoidalPositionalEncoding,
)
from cadgenesis.transformer.self_designing import SelfDesigningTransformer
from cadgenesis.transformer.transformer_block import CADTransformerBlock


@pytest.fixture
def mini_config() -> CADConfig:
    return CADConfig.mini()


class TestPositional:
    def test_sinusoidal_shape(self):
        pe = SinusoidalPositionalEncoding(d_model=128)
        x = torch.randn(2, 16, 128)
        out = pe(x)
        assert out.shape == (2, 16, 128)

    def test_rope_rotation(self):
        rope = RotaryEmbedding(dim=32)
        q = torch.randn(2, 4, 16, 32)
        k = torch.randn(2, 4, 16, 32)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_alibi_shape(self):
        alibi = ALiBiBias(num_heads=4)
        bias = alibi(seq_len=16, device=torch.device("cpu"))
        assert bias.shape == (1, 4, 16, 16)


class TestAttentionHeads:
    def test_self_attention(self):
        attn = SelfAttention(d_model=128, num_heads=4)
        x = torch.randn(2, 16, 128)
        out = attn(x)
        assert out.shape == (2, 16, 128)

    def test_geometry_attention(self):
        attn = GeometryAttention(d_model=128, num_heads=4)
        q = torch.randn(2, 16, 128)
        kv = torch.randn(2, 24, 128)
        out = attn(q, key_value=kv)
        assert out.shape == (2, 16, 128)

    def test_constraint_attention(self):
        attn = ConstraintAttention(d_model=128, num_heads=4)
        x = torch.randn(2, 16, 128)
        out = attn(x)
        assert out.shape == (2, 16, 128)

    def test_memory_attention(self):
        attn = MemoryAttention(d_model=128, num_heads=4)
        x = torch.randn(2, 16, 128)
        mem = torch.randn(2, 8, 128)
        out = attn(x, memory_bank=mem)
        assert out.shape == (2, 16, 128)

    def test_agent_attention(self):
        attn = AgentAttention(d_model=128, num_heads=4)
        x = torch.randn(2, 16, 128)
        out = attn(x)
        assert out.shape == (2, 16, 128)

    def test_uncertainty_attention(self):
        attn = UncertaintyAttention(d_model=128, num_heads=4)
        x = torch.randn(2, 16, 128)
        out, conf = attn(x)
        assert out.shape == (2, 16, 128)
        assert conf.shape == (2, 16, 1)

    def test_attention_mixture(self):
        mix = MultiHeadAttentionMixture(
            d_model=128,
            self_heads=2,
            geometry_heads=1,
            constraint_heads=0,
            memory_heads=0,
            agent_heads=1,
            uncertainty_heads=0,
        )
        x = torch.randn(2, 16, 128)
        out, _conf = mix(x)
        assert out.shape == (2, 16, 128)


class TestTransformerBlock:
    def test_block_forward(self):
        block = CADTransformerBlock(
            d_model=128,
            self_heads=2,
            geometry_heads=1,
            constraint_heads=0,
            memory_heads=0,
            agent_heads=1,
            uncertainty_heads=0,
            dim_feedforward=256,
        )
        x = torch.randn(2, 16, 128)
        out, _conf = block(x)
        assert out.shape == (2, 16, 128)


class TestFoundationModel:
    def test_full_model_forward(self, mini_config):
        model = GeometryAwareTransformer(mini_config)
        src = torch.randint(0, 50, (2, 16))
        tgt_in = torch.randint(0, 30, (2, 8))
        tgt_type = torch.randint(0, 3, (2, 8))

        logits, conf = model(src, tgt_in, tgt_type)
        assert logits.shape == (2, 8, model.cad_vocab_size)
        assert conf.shape == (2, 8, 1)

    def test_model_backward(self, mini_config):
        config = CADConfig.mini()
        config.model.memory_attn_heads = 1
        model = GeometryAwareTransformer(config)
        src = torch.randint(0, 50, (2, 16))
        tgt_in = torch.randint(0, 30, (2, 8))
        tgt_type = torch.randint(0, 3, (2, 8))

        logits, conf = model(src, tgt_in, tgt_type)
        reward = model.reward_model(model.encode(src))
        loss = logits.sum() + conf.sum() + reward.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_self_designing_wrapper(self, mini_config):
        wrapper = SelfDesigningTransformer(mini_config)
        src = torch.randint(0, 50, (2, 16))
        tgt_in = torch.randint(0, 30, (2, 8))
        tgt_type = torch.randint(0, 3, (2, 8))

        logits, _conf = wrapper(src, tgt_in, tgt_type)
        assert logits.shape[1] == 8
        score = wrapper.evaluate_complexity(src)
        assert score.shape == (2, 1)
