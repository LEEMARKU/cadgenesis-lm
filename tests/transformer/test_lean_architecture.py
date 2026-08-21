"""
Lean-architecture + production-preset tests (scale-up / de-over-engineering).

The default :class:`ModelConfig` must be a plain, production-oriented
encoder-decoder: no experimental subsystems, no exotic attention heads, and a
validated head layout (``total_heads == nhead``).  The experimental stack
remains available behind explicit flags (still exercised through
``CADConfig.mini()`` for backward compatibility).
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.config import CADConfig
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer


def _assert_lean(model: GeometryAwareTransformer):
    assert model.memory_system is None
    assert model.multi_agent_system is None
    assert model.neuro_symbolic_engine is None
    assert model.reward_model is None
    for block in (*model.encoder_blocks, *model.decoder_blocks):
        mix = block.attn_mixture
        assert mix.constraint_attn is None
        assert mix.memory_attn is None
        assert mix.agent_attn is None
        assert mix.uncertainty_attn is None
        assert mix.self_attn is not None
        assert mix.geometry_attn is not None


def test_default_config_is_lean():
    cfg = CADConfig()
    m = cfg.model
    assert m.use_multi_agent_system is False
    assert m.use_memory_system is False
    assert m.use_neuro_symbolic_reasoning is False
    assert m.use_rlaf_reward_model is False
    assert m.use_confidence_head is True
    # Exotic heads default to 0; standard heads sum to nhead.
    assert (
        m.constraint_attn_heads,
        m.memory_attn_heads,
        m.agent_attn_heads,
        m.uncertainty_attn_heads,
    ) == (0, 0, 0, 0)
    assert m.self_attn_heads + m.geometry_attn_heads == m.nhead


def test_lean_model_builds_forward_backward():
    torch.manual_seed(0)
    model = GeometryAwareTransformer(CADConfig())
    _assert_lean(model)
    src = torch.randint(0, 50, (2, 12))
    tgt = torch.randint(0, 30, (2, 6))
    tgt_type = torch.randint(0, 3, (2, 6))
    logits, conf = model(src, tgt, tgt_type)
    assert logits.shape == (2, 6, model.cad_vocab_size)
    assert conf.shape == (2, 6, 1)
    (logits.sum() + conf.sum()).backward()


def test_lean_model_kv_cache_equivalence():
    torch.manual_seed(0)
    model = GeometryAwareTransformer(CADConfig())
    model.eval()
    B, S, T = 2, 12, 6
    src = torch.randint(0, 50, (B, S))
    tgt = torch.randint(0, 30, (B, T))
    tgt_type = torch.randint(0, 3, (B, T))
    with torch.no_grad():
        full, _ = model(src, tgt, tgt_type)
        cache = model.prepare_decoder_cache(src)
        step_ids, step_tt = tgt[:, :1], tgt_type[:, :1]
        outs = []
        for i in range(T):
            step, _ = model.decode_step(step_ids, step_tt, cache)
            outs.append(step)
            if i + 1 < T:
                step_ids, step_tt = tgt[:, i + 1 : i + 2], tgt_type[:, i + 1 : i + 2]
        diff = (torch.cat(outs, 1) - full).abs().max().item()
    assert diff < 1e-4


@pytest.mark.parametrize(
    "preset,expected",
    [
        ("nano", (128, 4, 3, 3)),
        ("small", (384, 8, 6, 6)),
        ("base", (768, 12, 12, 12)),
        ("large", (1536, 16, 24, 24)),
    ],
)
def test_preset_scale_ladder(preset, expected):
    cfg = CADConfig.from_preset(preset)
    m = cfg.model
    assert (m.d_model, m.nhead, m.num_encoder_layers, m.num_decoder_layers) == expected
    assert m.self_attn_heads + m.geometry_attn_heads == m.nhead
    assert m.constraint_attn_heads == m.memory_attn_heads == 0
    assert m.agent_attn_heads == m.uncertainty_attn_heads == 0
    assert m.use_multi_agent_system is False
    assert m.use_memory_system is False


def test_preset_small_builds_and_decodes():
    torch.manual_seed(0)
    model = GeometryAwareTransformer(CADConfig.from_preset("small"))
    _assert_lean(model)
    model.eval()
    B, S, T = 2, 12, 6
    src = torch.randint(0, 50, (B, S))
    tgt = torch.randint(0, 30, (B, T))
    tgt_type = torch.randint(0, 3, (B, T))
    with torch.no_grad():
        full, _ = model(src, tgt, tgt_type)
        cache = model.prepare_decoder_cache(src)
        step_ids, step_tt = tgt[:, :1], tgt_type[:, :1]
        outs = []
        for i in range(T):
            step, _ = model.decode_step(step_ids, step_tt, cache)
            outs.append(step)
            if i + 1 < T:
                step_ids, step_tt = tgt[:, i + 1 : i + 2], tgt_type[:, i + 1 : i + 2]
        diff = (torch.cat(outs, 1) - full).abs().max().item()
    assert diff < 1e-4


def test_production_preset_is_lean_moe_mtp():
    cfg = CADConfig.production()
    m = cfg.model
    assert m.d_model == 1536
    assert m.use_moe is True
    assert m.mtp_depth == 1
    assert m.attention_backend == "gqa"
    assert m.use_multi_agent_system is False
    assert m.use_memory_system is False
    assert cfg.training.mixed_precision == "bf16"


def test_subsystem_flags_gate_heads():
    cfg = CADConfig()
    cfg.model.use_multi_agent_system = True
    cfg.model.agent_attn_heads = 1
    cfg.model.use_memory_system = True
    cfg.model.memory_attn_heads = 1
    cfg.model.use_neuro_symbolic_reasoning = True
    cfg.model.use_rlaf_reward_model = True
    # Rebalance nhead so validation still holds (each sub-attention also
    # requires its head count to divide d_model=1024).
    cfg.model.self_attn_heads = 4
    cfg.model.geometry_attn_heads = 8
    cfg.model.nhead = 16
    model = GeometryAwareTransformer(cfg)
    assert model.multi_agent_system is not None
    assert model.memory_system is not None
    assert model.neuro_symbolic_engine is not None
    assert model.reward_model is not None
    for block in (*model.encoder_blocks, *model.decoder_blocks):
        mix = block.attn_mixture
        assert mix.agent_attn is not None
        assert mix.memory_attn is not None


def test_agent_heads_require_subsystem_flag():
    cfg = CADConfig()
    cfg.model.agent_attn_heads = 2
    cfg.model.geometry_attn_heads = 6
    with pytest.raises(ValueError, match="use_multi_agent_system"):
        cfg._validate()


def test_memory_heads_require_subsystem_flag():
    cfg = CADConfig()
    cfg.model.memory_attn_heads = 2
    cfg.model.geometry_attn_heads = 6
    with pytest.raises(ValueError, match="use_memory_system"):
        cfg._validate()


def test_mini_keeps_full_research_stack():
    cfg = CADConfig.mini()
    assert cfg.model.use_multi_agent_system is True
    assert cfg.model.use_memory_system is True
    assert cfg.model.use_neuro_symbolic_reasoning is True
    assert cfg.model.use_rlaf_reward_model is True
    model = GeometryAwareTransformer(cfg)
    assert model.multi_agent_system is not None
    assert model.memory_system is not None
    assert model.neuro_symbolic_engine is not None
    assert model.reward_model is not None
    mix = model.decoder_blocks[0].attn_mixture
    assert mix.agent_attn is not None
