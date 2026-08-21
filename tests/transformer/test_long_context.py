"""
tests/transformer/test_long_context.py
======================================
Long-context support (v6.1 §4.7).

Before the fix, ``SinusoidalPositionalEncoding`` raised whenever a sequence
exceeded ``max_seq_len`` (2048 by default) — a hard context limit that
neither RoPE scaling nor ``max_position_embeddings`` could lift.  The table
is now grown on demand and the RoPE table always covers ``max_seq_len``.
"""

from __future__ import annotations

import math

import pytest
import torch

from cadgenesis.config import CADConfig
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer
from cadgenesis.transformer.positional import SinusoidalPositionalEncoding


def test_sinusoidal_encoding_grows_beyond_max_len():
    torch.manual_seed(0)
    enc = SinusoidalPositionalEncoding(d_model=32, max_len=16)
    x = torch.randn(1, 40, 32)  # 40 > 16: would previously raise
    out = enc(x)
    assert out.shape == x.shape
    # Values must match the closed-form sinusoidal formula at every position.
    expected = x + enc._build_table(40, 32).unsqueeze(0)
    assert torch.allclose(out, expected, atol=1e-6)


def test_sinusoidal_encoding_grows_with_position_offset():
    enc = SinusoidalPositionalEncoding(d_model=32, max_len=8)
    x = torch.randn(1, 4, 32)
    out = enc(x, position_offset=100)  # far beyond the initial table
    expected = x + enc._build_table(104, 32).unsqueeze(0)[:, 100:104]
    assert torch.allclose(out, expected, atol=1e-6)


def _long_model(seq_len: int) -> GeometryAwareTransformer:
    cfg = CADConfig.mini()
    cfg.model.max_seq_len = seq_len
    cfg.model.num_encoder_layers = 1
    cfg.model.num_decoder_layers = 1
    cfg.model.geometry_attn_heads = 0
    cfg.model.agent_attn_heads = 0
    cfg.model.use_multi_agent_system = False
    cfg.model.use_memory_system = False
    cfg.model.use_neuro_symbolic_reasoning = False
    cfg.model.use_rlaf_reward_model = False
    cfg.model.constraint_attn_heads = 0
    model = GeometryAwareTransformer(cfg)
    model.eval()
    return model


def test_model_forward_beyond_2048_context():
    """A sequence longer than the legacy 2048 limit must run and stay finite
    (also exercises the on-the-fly RoPE extension beyond its table)."""
    S, T = 4100, 4100
    model = _long_model(S)
    src = torch.randint(0, model.lang_vocab_size, (1, S))
    tgt = torch.randint(0, model.cad_vocab_size, (1, T))
    typ = torch.zeros_like(tgt)
    with torch.no_grad():
        logits, conf = model(src, tgt, typ)
    assert logits.shape == (1, T, model.cad_vocab_size)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(conf).all()


def test_rope_table_covers_max_seq_len():
    """The model's RoPE tables must be precomputed for at least max_seq_len."""
    model = _long_model(4096)
    rope = model.encoder_blocks[0].attn_mixture.self_attn.rope
    assert rope.max_position_embeddings >= 4096
    # And the growable sinusoidal table starts at max_seq_len.
    assert model.pos_enc.max_len == 4096