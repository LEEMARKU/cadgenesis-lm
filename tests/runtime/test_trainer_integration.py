"""
tests/runtime/test_trainer_integration.py
=========================================
HardwareAwareRuntime x trainer integration (v6.2): preset enforcement.
"""

from __future__ import annotations

import torch

from cadgenesis.config import CADConfig
from cadgenesis.tokenizer import AutonomousCADTokenizer
from cadgenesis.training.trainer import CADTrainer
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer


def _trainer(enforce: bool, preset: str = "gtx1650_4gb", batch: int = 256) -> CADTrainer:
    torch.manual_seed(0)
    cfg = CADConfig.mini()
    cfg.runtime.preset = preset
    cfg.runtime.enforce_preset = enforce
    cfg.training.batch_size = batch
    tok = AutonomousCADTokenizer.build_mini()
    tok.build_lang_vocab(["create a steel box"])
    model = GeometryAwareTransformer(cfg)
    return CADTrainer(cfg, model, tok, device="cpu")


def test_enforce_clamps_batch_size():
    tr = _trainer(enforce=True, batch=256)
    assert tr.config.training.batch_size == tr.runtime_preset.max_train_batch
    assert tr.runtime_preset.name == "gtx1650_4gb"


def test_no_enforce_keeps_user_batch():
    tr = _trainer(enforce=False, batch=256)
    assert tr.config.training.batch_size == 256


def test_enforce_does_not_grow_small_batch():
    tr = _trainer(enforce=True, batch=2)
    assert tr.config.training.batch_size == 2


def test_enforce_turns_on_checkpointing():
    tr = _trainer(enforce=True)
    assert tr.config.training.gradient_checkpointing is True


def test_preset_resolution_stored():
    tr = _trainer(enforce=False, preset="cpu")
    assert tr.runtime_preset.name == "cpu"