"""
tests/model/test_geometry_upgrade.py
====================================
Tests for the Geometry Transformer upgrade: geometry positional encoding,
feature interaction layers, efficient attention optimizations, and the
constraint-bias fix.  All tests are CPU-only and tiny.
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.config import CADConfig
from cadgenesis.transformer.attention import ConstraintAttention, MultiHeadAttentionMixture
from cadgenesis.transformer.efficient_attention import (
    BACKENDS,
    LinearAttention,
    SDPASelfAttention,
    build_self_attention,
)
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer
from cadgenesis.transformer.interaction import FeatureInteractionLayer
from cadgenesis.transformer.positional import GeometryPositionalEncoding
from cadgenesis.transformer.transformer_block import CADTransformerBlock


class TestGeometryPositionalEncoding:
    def test_shape_batched(self):
        pe = GeometryPositionalEncoding(d_model=64, use_fourier=True)
        x = torch.randn(2, 16, 64)
        coords = torch.randn(2, 16, 3)
        out = pe(x, coords)
        assert out.shape == (2, 16, 64)

    def test_shape_unbatched_coords(self):
        pe = GeometryPositionalEncoding(d_model=64, use_fourier=False)
        x = torch.randn(2, 16, 64)
        coords = torch.randn(16, 3)
        out = pe(x, coords)
        assert out.shape == (2, 16, 64)

    def test_none_coords_is_noop(self):
        pe = GeometryPositionalEncoding(d_model=64)
        x = torch.randn(2, 16, 64)
        out = pe(x, None)
        assert torch.equal(out, x)

    def test_fourier_feature_dim(self):
        pe = GeometryPositionalEncoding(d_model=64, use_fourier=True, num_frequencies=4)
        coords = torch.randn(3, 3)
        f = pe._features(coords)
        assert f.shape == (3, 3 * 2 * 4)

    def test_mismatched_coords_raises(self):
        pe = GeometryPositionalEncoding(d_model=64)
        x = torch.randn(2, 16, 64)
        with pytest.raises(ValueError):
            pe(x, torch.randn(2, 32, 3))

    def test_coord_wrong_last_dim_raises(self):
        pe = GeometryPositionalEncoding(d_model=64)
        with pytest.raises(ValueError):
            pe.embed(torch.randn(4, 5))

    def test_encoding_differs_with_coords(self):
        pe = GeometryPositionalEncoding(d_model=64)
        x = torch.zeros(1, 4, 64)
        a = pe(x, torch.zeros(1, 4, 3))
        b = pe(x, torch.ones(1, 4, 3))
        assert not torch.allclose(a, b)


class TestSDPASelfAttention:
    def test_shape_and_grad(self):
        attn = SDPASelfAttention(d_model=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == (2, 16, 64)
        out.sum().backward()
        assert attn.q_proj.weight.grad is not None

    def test_causal_mask(self):
        attn = SDPASelfAttention(d_model=64, num_heads=4)
        x = torch.randn(1, 8, 64)
        mask = torch.triu(torch.full((8, 8), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0)
        out = attn(x, attn_mask=mask)
        assert out.shape == (1, 8, 64)

    def test_bad_dims_raise(self):
        with pytest.raises(ValueError):
            SDPASelfAttention(d_model=64, num_heads=3)


class TestLinearAttention:
    def test_shape(self):
        attn = LinearAttention(d_model=64, num_heads=4, num_random_features=32)
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == (2, 16, 64)

    def test_causal_mask_is_implementable(self):
        attn = LinearAttention(d_model=32, num_heads=4, num_random_features=64)
        x = torch.randn(1, 8, 32)
        mask = torch.triu(torch.full((8, 8), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0)
        out = attn(x, attn_mask=mask)
        assert out.shape == (1, 8, 32)
        assert torch.isfinite(out).all()

    def test_nonneg_dense_and_finite(self):
        attn = LinearAttention(d_model=32, num_heads=4, num_random_features=64)
        x = torch.randn(2, 6, 32)
        out = attn(x)
        assert torch.isfinite(out).all()


class TestBuildAttentionFactory:
    @pytest.mark.parametrize("backend", ["math", "sdpa", "flash", "linear"])
    def test_all_backends_build_and_forward(self, backend):
        attn = build_self_attention(backend, d_model=64, num_heads=4, dropout=0.0)
        x = torch.randn(2, 8, 64)
        out = attn(x)
        assert out.shape == (2, 8, 64)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            build_self_attention("nystrom", d_model=64, num_heads=4)

    def test_backend_constants(self):
        assert set(BACKENDS) == {"math", "sdpa", "flash", "linear", "gqa", "mla"}

    def test_mixture_accepts_backend(self):
        mix = MultiHeadAttentionMixture(
            d_model=64,
            self_heads=2,
            geometry_heads=1,
            constraint_heads=0,
            memory_heads=0,
            agent_heads=0,
            uncertainty_heads=0,
            self_attn_backend="linear",
        )
        assert isinstance(mix.self_attn, LinearAttention)


class TestFeatureInteractionLayer:
    def test_shape(self):
        layer = FeatureInteractionLayer(d_model=64, num_heads=2)
        x = torch.randn(2, 16, 64)
        types = torch.randint(0, 5, (2, 16))
        out = layer(x, feature_type_ids=types)
        assert out.shape == (2, 16, 64)

    def test_backward(self):
        layer = FeatureInteractionLayer(d_model=64, num_heads=2)
        x = torch.randn(2, 16, 64)
        types = torch.randint(0, 5, (2, 16))
        out = layer(x, feature_type_ids=types)
        out.sum().backward()
        assert layer.type_bias.weight.grad is not None
        assert layer.gate.weight.grad is not None

    def test_causal_mask(self):
        layer = FeatureInteractionLayer(d_model=64, num_heads=2)
        x = torch.randn(1, 8, 64)
        mask = torch.triu(torch.full((8, 8), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0)
        out = layer(x, causal_mask=mask)
        assert out.shape == (1, 8, 64)


class TestConstraintBiasFix:
    def test_bias_proj_gradient_flows(self):
        attn = ConstraintAttention(d_model=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        out = attn(x)  # no mask → learned bias path
        assert out.shape == (2, 8, 64)
        out.sum().backward()
        assert attn.constraint_bias_proj.weight.grad is not None

    def test_mask_path_unchanged(self):
        attn = ConstraintAttention(d_model=64, num_heads=4)
        x = torch.randn(1, 8, 64)
        mask = torch.zeros(1, 1, 8, 8)
        out = attn(x, constraint_mask=mask)
        assert out.shape == (1, 8, 64)


class TestBlockAndModelUpgrade:
    def test_block_with_feature_interaction(self):
        block = CADTransformerBlock(
            d_model=64,
            self_heads=2,
            geometry_heads=1,
            constraint_heads=0,
            memory_heads=0,
            agent_heads=0,
            uncertainty_heads=0,
            dim_feedforward=128,
            use_feature_interaction=True,
            interaction_heads=1,
        )
        x = torch.randn(2, 8, 64)
        types = torch.randint(0, 4, (2, 8))
        out, _conf = block(x, feature_type_ids=types)
        assert out.shape == (2, 8, 64)

    def test_block_default_unchanged(self):
        block = CADTransformerBlock(d_model=64, dim_feedforward=128)
        assert block.feature_interaction is None
        assert block.attn_mixture.self_attn_backend == "math"

    @pytest.mark.parametrize("backend", ["math", "sdpa", "linear"])
    def test_full_model_upgrade(self, backend):
        cfg = CADConfig.mini()
        cfg.model.attention_backend = backend
        cfg.model.feature_interaction = True
        cfg.model.geometry_pos_encoding = True
        cfg.model.interaction_heads = 1
        model = GeometryAwareTransformer(cfg)

        src = torch.randint(0, 50, (2, 12))
        tgt_in = torch.randint(0, 30, (2, 8))
        tgt_type = torch.randint(0, 3, (2, 8))
        coords = torch.randn(2, 8, 3)

        logits, conf = model(src, tgt_in, tgt_type, geometry_coords=coords)
        assert logits.shape == (2, 8, model.cad_vocab_size)
        assert conf.shape == (2, 8, 1)

        (logits.sum() + conf.sum()).backward()
        assert model.encoder_blocks[0].attn_mixture.self_attn.q_proj.weight.grad is not None

    def test_config_validation(self):
        cfg = CADConfig.mini()
        cfg.model.attention_backend = "nystrom"
        with pytest.raises(ValueError):
            cfg._validate()
        cfg.model.attention_backend = "math"
        cfg.model.interaction_heads = 0
        with pytest.raises(ValueError):
            cfg._validate()

    def test_config_roundtrip(self, tmp_path):
        cfg = CADConfig.mini()
        cfg.model.attention_backend = "linear"
        cfg.model.feature_interaction = True
        cfg.model.geometry_pos_encoding = True
        path = tmp_path / "cfg.json"
        cfg.save(str(path))
        loaded = CADConfig.load(str(path))
        assert loaded.model.attention_backend == "linear"
        assert loaded.model.feature_interaction is True
        assert loaded.model.geometry_pos_encoding is True
