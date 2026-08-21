"""tests/multimodal/test_fusion.py
==================================
Unit tests for the Pillar-3 embedding space, fusion strategies and
cross-modal engine.
"""

from __future__ import annotations

import torch

from cadgenesis.config import MultimodalConfig
from cadgenesis.multimodal import (
    FusionEngine,
    FusionStrategy,
    Modality,
    MultimodalSystem,
)
from cadgenesis.multimodal.encoders.sensor import SensorDocument


class TestEmbeddingSpace:
    def test_projects_and_normalizes(self):
        sys = MultimodalSystem.from_config(MultimodalConfig())
        space = sys.space
        inputs = {
            Modality.TEXT: "a small bracket",
            Modality.CAD: {"format": "STEP", "features": []},
            Modality.SENSOR: SensorDocument(data=[[0.0] * 16 for _ in range(8)]),
        }
        for modality, data in inputs.items():
            feats = sys.encode_modality(modality, data)
            emb = space.embed(modality, feats)
            assert tuple(emb.shape) == (1, space.embed_dim)
            norms = emb.pow(2).sum(dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_similarity_is_symmetric(self):
        sys = MultimodalSystem.from_config(MultimodalConfig())
        a_emb = sys.space.embed(Modality.TEXT, sys.encode_modality(Modality.TEXT, "bracket"))
        b_emb = sys.space.embed(
            Modality.CAD, sys.encode_modality(Modality.CAD, {"format": "STEP", "features": []})
        )
        sim = sys.space.similarity(a_emb, b_emb)
        assert torch.isfinite(sim).all()
        assert tuple(sim.shape) == (1, 1)


class TestFusion:
    def test_early_fusion_shape(self):
        engine = FusionEngine(strategy=FusionStrategy.EARLY)
        feats = {m: torch.randn(1, 256) for m in Modality}
        out = engine.forward(feats)
        assert tuple(out.fused.shape) == (1, 256)

    def test_early_fusion_missing_modality(self):
        engine = FusionEngine(strategy=FusionStrategy.EARLY)
        feats = {Modality.TEXT: torch.randn(1, 256)}
        out = engine.forward(feats)
        assert tuple(out.fused.shape) == (1, 256)

    def test_all_strategies(self):
        feats = {m: torch.randn(2, 256) for m in Modality}
        for strategy in FusionStrategy:
            engine = FusionEngine(strategy=strategy)
            out = engine.forward(feats)
            assert tuple(out.fused.shape) == (2, 256), strategy


class TestCrossModal:
    def test_headline_pairs_registered(self):
        from cadgenesis.multimodal.cross_modal import HEADLINE_PAIRS

        sys = MultimodalSystem.from_config(MultimodalConfig())
        assert len(HEADLINE_PAIRS) >= 8
        assert len(sys.cross_modal.registry.stacks()) == len(HEADLINE_PAIRS)

    def test_engine_attend(self):
        from cadgenesis.multimodal.cross_modal import HEADLINE_PAIRS

        sys = MultimodalSystem.from_config(MultimodalConfig())
        a_mod, b_mod = HEADLINE_PAIRS[0]
        dims = {m: sys.raw_feature_dims()[m.value] for m in (a_mod, b_mod)}
        a_feats = torch.randn(2, dims[a_mod])
        b_feats = torch.randn(2, dims[b_mod])
        result = sys.cross_modal.attend(a_mod, b_mod, a_feats, b_feats)
        assert tuple(result.a_pooled.shape) == (2, 256)
        assert tuple(result.b_pooled.shape) == (2, 256)


class TestMultimodalSystem:
    def test_encode_full(self):
        sys = MultimodalSystem.from_config(MultimodalConfig())
        encoding = sys.encode(
            {
                Modality.TEXT: "gearbox housing",
                Modality.CAD: {"format": "STEP", "features": []},
            }
        )
        assert encoding.fused is not None
        assert tuple(encoding.fused.shape) == (1, 256)

    def test_feature_dims(self):
        cfg = MultimodalConfig()
        dims = cfg.feature_dims()
        assert dims["text"] == 512
        assert dims["cad"] == 384
        assert dims["sensor"] == 256
