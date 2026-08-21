"""tests/transformer/test_facades.py
===================================
Unit tests for the transformer facade/shim modules.
"""

from __future__ import annotations

import torch

from cadgenesis.transformer import positional_encoding as pos_shim
from cadgenesis.transformer import transformer as facade
from cadgenesis.transformer.constraint_attention import ConstraintAttention
from cadgenesis.transformer.expert_router import ExpertRouter
from cadgenesis.transformer.geometry_attention import GeometryAttention
from cadgenesis.transformer.layer_router import DynamicLayerRouter
from cadgenesis.transformer.memory_attention import MemoryAttention
from cadgenesis.transformer.uncertainty_attention import UncertaintyAttention


class TestFacade:
    def test_core_classes_present(self):
        for name in [
            "GeometryAwareTransformer",
            "CADTransformerBlock",
            "SelfDesigningTransformer",
            "EncoderStack",
            "DecoderStack",
            "CombinedInputEmbedding",
            "LMHead",
            "ConfidenceHead",
            "OutputHeads",
            "CADSequenceLoss",
            "ConfidenceLoss",
            "MaskedCrossEntropyLoss",
        ]:
            assert hasattr(facade, name), name

    def test_all_list_complete(self):
        for name in facade.__all__:
            assert hasattr(facade, name), name


class TestPositionalEncodingShim:
    def test_classes_present(self):
        for name in [
            "SinusoidalPositionalEncoding",
            "RotaryEmbedding",
            "ALiBiBias",
            "GeometryPositionalEncoding",
        ]:
            assert hasattr(pos_shim, name), name

    def test_alias_is_same_class(self):
        from cadgenesis.transformer.positional import RotaryEmbedding

        assert pos_shim.RotaryEmbedding is RotaryEmbedding


class TestAttentionShims:
    def test_geometry(self):
        attn = GeometryAttention(d_model=32, num_heads=2)
        out = attn(torch.randn(2, 8, 32), key_value=torch.randn(2, 10, 32))
        assert out.shape == (2, 8, 32)

    def test_constraint(self):
        attn = ConstraintAttention(d_model=32, num_heads=2)
        assert attn(torch.randn(2, 8, 32)).shape == (2, 8, 32)

    def test_memory(self):
        attn = MemoryAttention(d_model=32, num_heads=2)
        out = attn(torch.randn(2, 8, 32), memory_bank=torch.randn(2, 4, 32))
        assert out.shape == (2, 8, 32)

    def test_uncertainty(self):
        attn = UncertaintyAttention(d_model=32, num_heads=2)
        out, conf = attn(torch.randn(2, 8, 32))
        assert out.shape == (2, 8, 32)
        assert conf.shape == (2, 8, 1)


class TestExpertRouter:
    def test_route_shapes(self):
        router = ExpertRouter(d_model=16, num_experts=4, top_k=2)
        flat = torch.randn(10, 16)
        weights, idx = router(flat)
        assert weights.shape == (10, 2)
        assert idx.shape == (10, 2)
        assert (idx.max() < 4).item()

    def test_weights_normalised(self):
        router = ExpertRouter(d_model=16, num_experts=4, top_k=2)
        weights, _ = router(torch.randn(10, 16), use_jitter=False)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(10), atol=1e-5)

    def test_load_balance_loss(self):
        router = ExpertRouter(d_model=16, num_experts=4, top_k=2)
        flat = torch.randn(10, 16)
        probs = torch.softmax(router.linear(flat), dim=-1)
        _, idx = router(flat, use_jitter=False)
        loss = router.load_balance_loss(probs, idx)
        assert loss.item() > 0

    def test_grow(self):
        router = ExpertRouter(d_model=16, num_experts=2, top_k=2)
        router.grow(n=2)
        assert router.num_experts == 4
        assert router.linear.weight.shape == (4, 16)

    def test_validation(self):
        import pytest

        with pytest.raises(ValueError):
            ExpertRouter(d_model=16, num_experts=2, top_k=3)
        with pytest.raises(ValueError):
            ExpertRouter(d_model=16, num_experts=0, top_k=0)


class TestLayerRouterShim:
    def test_present(self):
        assert DynamicLayerRouter is not None
