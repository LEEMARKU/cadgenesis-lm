"""
tests/runtime/test_config_runtime.py
====================================
RuntimeConfig serialization round-trip through CADConfig (v6.2).
"""

from __future__ import annotations

from cadgenesis.config import CADConfig
from cadgenesis.config.cad_config import RuntimeConfig


def test_runtime_config_defaults():
    cfg = CADConfig()
    assert isinstance(cfg.runtime, RuntimeConfig)
    assert cfg.runtime.preset == "auto"
    assert cfg.runtime.enforce_preset is False


def test_runtime_config_round_trip():
    cfg = CADConfig()
    cfg.runtime.preset = "gtx1650_4gb"
    cfg.runtime.enforce_preset = True

    raw = cfg.to_dict()
    cfg2 = CADConfig.from_dict(raw)
    assert cfg2.runtime.preset == "gtx1650_4gb"
    assert cfg2.runtime.enforce_preset is True


def test_runtime_config_survives_mini():
    cfg = CADConfig.mini()
    cfg.runtime.preset = "cpu"
    raw = cfg.to_dict()
    cfg2 = CADConfig.from_dict(raw)
    assert cfg2.runtime.preset == "cpu"


def test_runtime_config_unknown_preset_passes_through_config():
    """Config stores the string; the runtime package validates it on use."""
    cfg = CADConfig()
    cfg.runtime.preset = "gtx1650_4gb"
    assert cfg._validate() is None