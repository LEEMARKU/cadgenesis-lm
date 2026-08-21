"""
tests/config/test_mla_small_models.py
=====================================
MLA configuration validation for small models (v6.1 §4.6).

Small configs (e.g. ``mini``: d_model=128, self_attn_heads=2 → head_dim 64)
carry the default ``qk_rope_head_dim=64`` which exceeds head_dim.  The
validation must auto-clamp to a working even value instead of failing, and
the resulting model must run a finite forward — both when the config is
constructed with MLA enabled and when it is mutated afterwards.
"""

from __future__ import annotations

import torch

from cadgenesis.config import CADConfig, ModelConfig
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer


def _assert_mla_runs(cfg: CADConfig) -> None:
    head_dim = cfg.model.d_model // cfg.model.self_attn_heads
    if cfg.model.qk_rope_head_dim >= head_dim:
        # Post-construction mutation: the clamp happens at model build time
        # (_block_kwargs), so only the *effective* rope dim is guaranteed.
        assert cfg.model.qk_rope_head_dim == 64  # the untouched default
    else:
        assert cfg.model.qk_rope_head_dim % 2 == 0, (
            "clamped qk_rope_head_dim must stay even (RoPE requires an even dim)"
        )
    model = GeometryAwareTransformer(cfg)
    rope = model.encoder_blocks[0].attn_mixture.self_attn.rope
    assert rope.dim < head_dim, f"effective rope dim {rope.dim} >= head_dim {head_dim}"
    assert rope.dim % 2 == 0, "effective rope dim must be even"
    model.eval()
    src = torch.randint(0, 400, (1, 8))
    tgt = torch.randint(0, 400, (1, 8))
    typ = torch.zeros_like(tgt)
    with torch.no_grad():
        logits, _ = model(src, tgt, typ)
    assert torch.isfinite(logits).all()


def test_mla_mini_config_constructed_with_mla():
    """CADConfig.mini() + MLA at construction time must auto-clamp."""
    cfg = CADConfig.mini()
    cfg.model.attention_backend = "mla"
    # Force re-validation the way serialization round-trips do.
    rebuilt = CADConfig.from_dict(cfg.to_dict())
    assert rebuilt.model.attention_backend == "mla"
    _assert_mla_runs(rebuilt)


def test_mla_mini_config_mutated_after_construction():
    """Post-construction mutation bypasses _validate; the model must still
    clamp via _block_kwargs at build time."""
    cfg = CADConfig.mini()
    cfg.model.attention_backend = "mla"
    _assert_mla_runs(cfg)


def test_mla_tiny_model_from_scratch():
    """A tiny hand-built MLA config must clamp and run."""
    cfg = CADConfig(
        model=ModelConfig(
            d_model=32,
            nhead=2,
            self_attn_heads=2,
            geometry_attn_heads=0,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            attention_backend="mla",
            kv_lora_rank=16,
        )
    )
    _assert_mla_runs(cfg)


def test_mla_default_large_config_unchanged():
    """A big model whose head_dim exceeds the default qk_rope_head_dim must
    keep its configured value untouched."""
    cfg = CADConfig.from_preset("base")
    cfg.model.attention_backend = "mla"
    rebuilt = CADConfig.from_dict(cfg.to_dict())
    assert rebuilt.model.qk_rope_head_dim == 64  # head_dim = 768/8 = 96 > 64
    _assert_mla_runs(rebuilt)