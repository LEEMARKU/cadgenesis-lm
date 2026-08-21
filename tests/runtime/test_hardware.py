"""
tests/runtime/test_hardware.py
==============================
HardwareAwareRuntime (v6.2) — preset selection, detection and clamping.
"""

from __future__ import annotations

import pytest

from cadgenesis.runtime.hardware import (
    PRESETS,
    RuntimePreset,
    clamp_to_preset,
    detect_device,
    select_preset,
)


def test_all_presets_are_declared():
    assert {"gtx1650_4gb", "rtx3050_8gb", "cpu"} <= set(PRESETS)


def test_gtx1650_preset_values():
    p = PRESETS["gtx1650_4gb"]
    assert p.vram_mb == 4095
    assert p.compute_cap == (7, 5)
    assert p.max_train_batch == 8
    assert p.max_seq_len == 2048
    assert p.grad_checkpointing is True


def test_bf16_support_requires_compute_80():
    assert PRESETS["gtx1650_4gb"].supports_bf16 is False
    assert PRESETS["rtx3050_8gb"].supports_bf16 is True
    assert PRESETS["cpu"].supports_bf16 is False


def test_select_named_preset():
    assert select_preset("gtx1650_4gb") is PRESETS["gtx1650_4gb"]
    assert select_preset("CPU") is PRESETS["cpu"]


def test_select_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown runtime preset"):
        select_preset("bogus")


def test_select_auto_returns_valid_preset(monkeypatch):
    monkeypatch.delenv("CADGENESIS_RUNTIME_PRESET", raising=False)
    p = select_preset("auto")
    assert isinstance(p, RuntimePreset)


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("CADGENESIS_RUNTIME_PRESET", "rtx3050_8gb")
    assert select_preset() is PRESETS["rtx3050_8gb"]
    assert select_preset(None) is PRESETS["rtx3050_8gb"]


def test_detect_device_shape():
    kind, vram_mb = detect_device()
    assert kind in ("cuda", "cpu")
    assert vram_mb >= 0


def test_clamp_to_preset():
    p = PRESETS["gtx1650_4gb"]
    out = clamp_to_preset(p, train_batch=128, eval_batch=64, max_seq_len=4096, other=7)
    assert out == {"train_batch": 8, "eval_batch": 16, "max_seq_len": 2048, "other": 7}


def test_clamp_never_increases():
    p = PRESETS["gtx1650_4gb"]
    out = clamp_to_preset(p, train_batch=2, max_seq_len=256)
    assert out == {"train_batch": 2, "max_seq_len": 256}