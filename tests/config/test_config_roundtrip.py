"""Tests for the config <-> dict round-trip (checkpoint reproducibility)."""

from cadgenesis.config import CADConfig


def test_from_dict_roundtrip_mini():
    cfg = CADConfig.mini()
    cfg.training.lr = 1e-3
    cfg.experiment_name = "roundtrip"
    restored = CADConfig.from_dict(cfg.to_dict())
    assert restored.to_dict() == cfg.to_dict()


def test_from_dict_roundtrip_preset_small():
    cfg = CADConfig.from_preset("small")
    restored = CADConfig.from_dict(cfg.to_dict())
    assert restored.to_dict() == cfg.to_dict()
    assert restored.model.d_model == cfg.model.d_model


def test_from_dict_rejects_invalid():
    bad = CADConfig.mini().to_dict()
    bad["training"]["mixed_precision"] = "exotic"
    import pytest

    with pytest.raises(ValueError):
        CADConfig.from_dict(bad)


def test_from_dict_missing_sections_use_defaults():
    restored = CADConfig.from_dict({})
    assert restored.model.d_model == CADConfig().model.d_model
