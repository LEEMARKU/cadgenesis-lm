"""tests/transformer/test_hierarchical_transformer.py
======================================================
Unit tests for the five-stage hierarchical CAD transformer (Pillar 1).
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.config import CADConfig
from cadgenesis.transformer.hierarchical_transformer import (
    STAGE_NAMES,
    HierarchicalCADTransformer,
)


def _hier_config(**model_overrides) -> CADConfig:
    cfg = CADConfig.mini()
    cfg.model.use_hierarchical_transformer = True
    for key, value in model_overrides.items():
        setattr(cfg.model, key, value)
    return cfg


def _inputs(model, B=2, S=12, T=8):
    src = torch.randint(0, 50, (B, S))
    tgt_in = torch.randint(0, 30, (B, T))
    tgt_type = torch.randint(0, 3, (B, T))
    return src, tgt_in, tgt_type


class TestHierarchicalStructure:
    def test_stage_names(self):
        assert STAGE_NAMES == ("planner", "geometry", "constraint", "execution", "validation")

    def test_stage_blocks_created(self):
        cfg = _hier_config()
        model = HierarchicalCADTransformer(cfg)
        for stage in STAGE_NAMES:
            assert len(getattr(model, f"{stage}_blocks")) == 1
            assert hasattr(model, f"{stage}_norm")

    def test_custom_depths(self):
        cfg = _hier_config(geometry_layers=2, execution_layers=3)
        model = HierarchicalCADTransformer(cfg)
        assert len(model.geometry_blocks) == 2
        assert len(model.execution_blocks) == 3
        # Decoder stages only (the planner is the encoder).
        assert model.total_decoder_layers == 2 + 1 + 3 + 1

    def test_zero_stage_raises(self):
        cfg = _hier_config(geometry_layers=0)
        with pytest.raises(ValueError):
            HierarchicalCADTransformer(cfg)


class TestForwardContract:
    def test_output_shapes(self):
        model = HierarchicalCADTransformer(_hier_config())
        src, tgt_in, tgt_type = _inputs(model)
        logits, conf = model(src, tgt_in, tgt_type)
        assert logits.shape == (2, 8, model.cad_vocab_size)
        assert conf.shape == (2, 8, 1)

    def test_gradients_flow(self):
        model = HierarchicalCADTransformer(_hier_config())
        src, tgt_in, tgt_type = _inputs(model)
        logits, conf = model(src, tgt_in, tgt_type)
        (logits.sum() + conf.sum()).backward()
        assert model.planner_blocks[0].attn_mixture.self_attn.q_proj.weight.grad is not None
        assert model.out_proj.weight.grad is not None

    def test_geometry_coords(self):
        cfg = _hier_config(geometry_pos_encoding=True)
        model = HierarchicalCADTransformer(cfg)
        src, tgt_in, tgt_type = _inputs(model)
        coords = torch.randn(2, 8, 3)
        logits, _ = model(src, tgt_in, tgt_type, geometry_coords=coords)
        assert logits.shape == (2, 8, model.cad_vocab_size)

    def test_constraint_mask_accepted(self):
        model = HierarchicalCADTransformer(_hier_config())
        src, tgt_in, tgt_type = _inputs(model, T=8)
        mask = torch.zeros(1, 1, 8, 8)
        logits, _ = model(src, tgt_in, tgt_type, constraint_mask=mask)
        assert logits.shape[1] == 8

    def test_memory_bank_accepted(self):
        model = HierarchicalCADTransformer(_hier_config())
        src, tgt_in, tgt_type = _inputs(model)
        bank = torch.randn(2, 16, model.d_model)
        logits, _ = model(src, tgt_in, tgt_type, memory_bank=bank)
        assert logits.shape[1] == 8


class TestDynamicRoutingIntegration:
    def test_budget_limits_layers(self):
        cfg = _hier_config(computation_budget=0.3)  # 4 decoder layers -> cap 2
        model = HierarchicalCADTransformer(cfg)
        assert model.routing.max_layers == 2
        src, tgt_in, tgt_type = _inputs(model)
        model(src, tgt_in, tgt_type)
        report = model.routing.report()
        assert report["layers_executed"] <= 2
        assert report["savings_fraction"] >= 0.5

    def test_early_exit_by_confidence(self):
        cfg = _hier_config(early_exit_threshold=0.999)
        model = HierarchicalCADTransformer(cfg)
        src, tgt_in, tgt_type = _inputs(model)
        logits, _ = model(src, tgt_in, tgt_type)
        # threshold is extreme; the model will almost certainly complete.
        assert logits.shape[1] == 8
        assert model.routing.report()["exit_reason"] in ("budget", "completed")


class TestSpecializedMoEIntegration:
    def test_moe_blocks_injected(self):
        cfg = _hier_config(use_specialized_moe=True)
        model = HierarchicalCADTransformer(cfg)
        for stage in STAGE_NAMES:
            block = getattr(model, f"{stage}_blocks")[0]
            assert block.moe_layer() is not None
            assert block.is_moe

    def test_moe_forward_and_aux(self):
        cfg = _hier_config(use_specialized_moe=True)
        model = HierarchicalCADTransformer(cfg)
        src, tgt_in, tgt_type = _inputs(model)
        logits, conf = model(src, tgt_in, tgt_type)
        (logits.sum() + conf.sum() + model.aux_loss()).backward()
        assert float(model.aux_loss().item()) > 0

    def test_stage_report_expert_loads(self):
        cfg = _hier_config(use_specialized_moe=True)
        model = HierarchicalCADTransformer(cfg)
        src, tgt_in, tgt_type = _inputs(model)
        model(src, tgt_in, tgt_type)
        report = model.stage_report()
        assert set(report["stages"]) == set(STAGE_NAMES)
        assert "expert_load" in report
        assert report["routing"]["total_layers"] == model.total_decoder_layers

    def test_dense_stage_report(self):
        model = HierarchicalCADTransformer(_hier_config())
        report = model.stage_report()
        assert "expert_load" not in report
        assert report["routing"]["total_layers"] == model.total_decoder_layers
