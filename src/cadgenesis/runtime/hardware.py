"""
cadgenesis.runtime.hardware
===========================
Hardware detection and device presets (v6.2 HardwareAwareRuntime).

Every model/batch decision that depends on the machine's GPU/CPU capabilities
is derived from here instead of hardcoded constants.  Presets are *lower
bounds of available memory*, so a machine can always over-ride via
``CADConfig.runtime.preset``.

Supported presets:

* ``gtx1650_4gb`` — NVIDIA GeForce GTX 1650 (4095 MiB, compute 7.5): the
  reference machine this repository is developed on.
* ``rtx3050_8gb`` — NVIDIA GeForce RTX 3050 (8 GB, compute 8.6).
* ``cpu`` — no CUDA device; CPU-only training/inference budgets.
* ``auto`` — select by live detection (torch + torch.cuda).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

PRESETS: dict[str, RuntimePreset] = {}


@dataclass(frozen=True)
class RuntimePreset:
    """Memory/throughput budget for a device class."""

    name: str
    vram_mb: int | None
    compute_cap: tuple[int, int] | None
    max_train_batch: int
    max_eval_batch: int
    max_seq_len: int
    dtype: str
    grad_checkpointing: bool
    max_model_params: int
    cpu_threads: int
    notes: str = ""

    @property
    def supports_bf16(self) -> bool:
        """bf16 autocast is only useful on compute capability >= 8.0 GPUs."""
        return bool(self.compute_cap and self.compute_cap >= (8, 0))


def _register(preset: RuntimePreset) -> RuntimePreset:
    PRESETS[preset.name] = preset
    return preset


_gtx1650_4gb = _register(
    RuntimePreset(
        name="gtx1650_4gb",
        vram_mb=4095,
        compute_cap=(7, 5),
        max_train_batch=8,
        max_eval_batch=16,
        max_seq_len=2048,
        dtype="fp16",
        grad_checkpointing=True,
        max_model_params=48_000_000,
        cpu_threads=4,
        notes=(
            "GTX 1650 (compute 7.5): no bf16 tensor cores; use fp16 with a "
            "grad scaler or fp32; gradient checkpointing on for training; "
            "small-batch budget."
        ),
    )
)

_rtx3050_8gb = _register(
    RuntimePreset(
        name="rtx3050_8gb",
        vram_mb=8192,
        compute_cap=(8, 6),
        max_train_batch=16,
        max_eval_batch=32,
        max_seq_len=4096,
        dtype="bf16",
        grad_checkpointing=False,
        max_model_params=120_000_000,
        cpu_threads=8,
        notes=(
            "RTX 3050 (compute 8.6): bf16 capable, double the 1650's VRAM; "
            "checkpointing only needed past ~120M params."
        ),
    )
)

_cpu = _register(
    RuntimePreset(
        name="cpu",
        vram_mb=None,
        compute_cap=None,
        max_train_batch=32,
        max_eval_batch=64,
        max_seq_len=2048,
        dtype="bf16",
        grad_checkpointing=False,
        max_model_params=200_000_000,
        cpu_threads=os.cpu_count() or 4,
        notes="CPU-only fallback; bf16 autocast supported on modern CPUs.",
    )
)


def detect_device() -> tuple[str, int]:
    """Return ``("cuda", vram_mb)`` or ``("cpu", 0)`` by live probing."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return "cuda", int(props.total_memory // (1024 * 1024))
    return "cpu", 0


def select_preset(name: str | None = None) -> RuntimePreset:
    """
    Resolve a preset name, honouring ``CADGENESIS_RUNTIME_PRESET`` env var.

    ``None``/``"auto"`` performs live detection and maps it to the closest
    declared preset (best-effort), falling back to ``cpu``.
    """
    name = (name or os.environ.get("CADGENESIS_RUNTIME_PRESET") or "auto").lower()
    if name in PRESETS:
        return PRESETS[name]
    if name != "auto":
        raise ValueError(
            f"Unknown runtime preset {name!r}; "
            f"choose one of {sorted(PRESETS)} or 'auto'."
        )

    kind, vram_mb = detect_device()
    if kind != "cuda":
        return _cpu
    if vram_mb <= 6144:  # < 6 GB → 4 GB class (1650)
        return _gtx1650_4gb
    if vram_mb <= 10_240:  # 6-10 GB -> 8 GB class (3050)
        return _rtx3050_8gb
    return _rtx3050_8gb  # larger cards still fit the 8 GB budget initially


def clamp_to_preset(preset: RuntimePreset, **values: int) -> dict[str, int]:
    """
    Clamp free-form ints (batch sizes, seq len, ...) to a preset's ceilings.

    Unknown keys pass through unchanged.  Useful for configs that carry
    explicit user values but must respect the device budget.
    """
    bounds: dict[str, int] = {
        "train_batch": preset.max_train_batch,
        "eval_batch": preset.max_eval_batch,
        "max_seq_len": preset.max_seq_len,
    }
    out: dict[str, int] = {}
    for key, value in values.items():
        if key in bounds:
            out[key] = min(value, bounds[key])
        else:
            out[key] = value
    return out
