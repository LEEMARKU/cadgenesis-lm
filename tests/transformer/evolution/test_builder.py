"""tests/transformer/evolution/test_builder.py
================================================
Unit tests for the configuration-driven architecture builder.
"""

from __future__ import annotations

import json

import pytest
import torch

from cadgenesis.config import CADConfig
from cadgenesis.transformer.evolution.builder import (
    ConfigurationDrivenBuilder,
    RegistryStack,
)
from cadgenesis.transformer.hierarchical_transformer import HierarchicalCADTransformer

MINI_ARCH = {
    "type": "standard",
    "name": "mini-arch",
    "d_model": 128,
    "nhead": 4,
    "heads": {"self": 2, "geometry": 1, "agent": 1},
    "encoder_layers": 1,
    "decoder_layers": 1,
}


class TestConfigurationDrivenBuilder:
    def test_build_standard(self):
        builder = ConfigurationDrivenBuilder()
        model = builder.build_model(MINI_ARCH)
        assert model.__class__.__name__ == "GeometryAwareTransformer"
        src = torch.randint(0, 50, (2, 8))
        tgt_in = torch.randint(0, 30, (2, 6))
        tgt_type = torch.randint(0, 3, (2, 6))
        logits, _ = model(src, tgt_in, tgt_type)
        assert logits.shape[1] == 6

    def test_build_hierarchical(self):
        arch = dict(
            MINI_ARCH,
            type="hierarchical",
            stages={"planner": 1, "geometry": 1, "constraint": 1, "execution": 1, "validation": 1},
        )
        builder = ConfigurationDrivenBuilder()
        model = builder.build_model(arch)
        assert isinstance(model, HierarchicalCADTransformer)

    def test_build_stack(self):
        spec = {
            "type": "stack",
            "layers": [
                {"kind": "rms_norm", "params": {"dim": 64}},
                {"kind": "self_attention", "params": {"d_model": 64, "num_heads": 4}},
                {"kind": "swiglu_ffn", "params": {"d_model": 64, "dim_feedforward": 128}},
            ],
        }
        builder = ConfigurationDrivenBuilder()
        stack = builder.build_model(spec)
        assert isinstance(stack, RegistryStack)
        out = stack(torch.randn(2, 8, 64))
        assert out.shape == (2, 8, 64)

    def test_validate_unknown_kind(self):
        builder = ConfigurationDrivenBuilder()
        with pytest.raises(KeyError):
            builder.build_model({"type": "stack", "layers": [{"kind": "nope"}]})

    def test_validate_unknown_type(self):
        builder = ConfigurationDrivenBuilder()
        with pytest.raises(ValueError):
            builder.build_model({"type": "impossible"})

    def test_build_from_config_dispatch(self):
        cfg = CADConfig.mini()
        builder = ConfigurationDrivenBuilder(config=cfg)
        assert builder.build_from_config().__class__.__name__ == "GeometryAwareTransformer"
        cfg.model.use_hierarchical_transformer = True
        assert builder.build_from_config(cfg).__class__.__name__ == "HierarchicalCADTransformer"

    def test_heads_must_sum_to_nhead(self):
        builder = ConfigurationDrivenBuilder()
        bad = dict(MINI_ARCH, heads={"self": 2, "geometry": 2})
        bad["nhead"] = 8
        with pytest.raises(ValueError):
            builder.build_model(bad)

    def test_spec_roundtrip_json(self):
        arch = json.loads(json.dumps(MINI_ARCH))
        builder = ConfigurationDrivenBuilder()
        assert builder.build_model(arch) is not None

    def test_describe(self):
        builder = ConfigurationDrivenBuilder()
        plan = builder.describe(MINI_ARCH)
        assert plan["type"] == "standard"
        assert plan["name"] == "mini-arch"

    def test_ffn_kinds_resolved(self):
        arch = dict(MINI_ARCH, ffn="specialized_moe", attention="sparse")
        builder = ConfigurationDrivenBuilder()
        model = builder.build_model(arch)
        assert model.__class__.__name__ == "GeometryAwareTransformer"

    def test_empty_stack_rejected(self):
        builder = ConfigurationDrivenBuilder()
        with pytest.raises(ValueError):
            builder.build_model({"type": "stack", "layers": []})
