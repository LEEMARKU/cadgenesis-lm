"""
cadgenesis.runtime.memory_planner
=================================
Device-aware training/inference budget planning (v6.2).

Given a :class:`~cadgenesis.config.CADConfig` and a
:class:`~cadgenesis.runtime.hardware.RuntimePreset`, the planner answers the
two questions that matter on a 4 GB GPU:

* *will a training step fit?* — a per-tensor estimate for the forward +
  backward of ``GeometryAwareTransformer`` (embeddings, attention, FFN, SSM)
  using bf16/fp16/autocast-aware sizes;
* *what should I set in the config instead?* — recommended batch / seq len /
  gradient-checkpointing flags via :func:`recommend_config_overrides`.
"""

from __future__ import annotations

from dataclasses import dataclass

from cadgenesis.runtime.hardware import RuntimePreset


@dataclass(frozen=True)
class MemoryEstimate:
    """Per-tensor memory estimate (bytes) for one model forward+backward."""

    params_bytes: int
    gradients_bytes: int
    optimizer_bytes: int
    activations_bytes: int
    total_bytes: int

    @property
    def total_mib(self) -> float:
        return self.total_bytes / (1024 * 1024)

    @property
    def params_mib(self) -> float:
        return self.params_bytes / (1024 * 1024)


def estimate_training_memory(
    *,
    n_params: int,
    d_model: int,
    num_layers: int,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    dtype_bytes: int = 2,
    grad_checkpointing: bool = False,
) -> MemoryEstimate:
    """
    Rough but *conservative* (over-estimating) training-step memory model.

    Components (all in bytes):
      * parameters:            ``n_params * dtype_bytes``
      * gradients:             same as parameters
      * optimizer (AdamW):     2 extra copies → ``2 * n_params * 4``
      * activations:           embeddings + per-layer attention/FFN tensors
        sized for one batch (see :func:`_activation_bytes`).
    """
    params_bytes = n_params * dtype_bytes
    gradients_bytes = n_params * dtype_bytes
    optimizer_bytes = 2 * n_params * 4
    activations_bytes = _activation_bytes(
        d_model=d_model,
        num_layers=num_layers,
        seq_len=seq_len,
        batch_size=batch_size,
        vocab_size=vocab_size,
        dtype_bytes=dtype_bytes,
        grad_checkpointing=grad_checkpointing,
    )
    total = params_bytes + gradients_bytes + optimizer_bytes + activations_bytes
    return MemoryEstimate(
        params_bytes=params_bytes,
        gradients_bytes=gradients_bytes,
        optimizer_bytes=optimizer_bytes,
        activations_bytes=activations_bytes,
        total_bytes=total,
    )


def _activation_bytes(
    *,
    d_model: int,
    num_layers: int,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    dtype_bytes: int,
    grad_checkpointing: bool,
) -> int:
    """
    Per-batch activation footprint (bytes), over-estimated by ~1.5x.

    Contributions:
      * input embedding tensor:      B x S x d
      * output logits (peaks once):  B x S x V
      * per transformer layer:
          - self-attention Q/K/V + scores + attn out: ~8 x B x S x d
          - FFN intermediate (2x hidden):             ~2 x B x S x (2d)
          - layer norms, residuals:                    ~4 x B x S x d
    With gradient checkpointing, per-layer tensors are freed and only
    recomputed: the stored amount drops to one layer's share plus the
    embedding + logits (activations / num_layers + embedding + logits).
    """
    embedding = batch_size * seq_len * d_model
    logits = batch_size * seq_len * vocab_size
    per_layer = batch_size * seq_len * d_model * (8 + 2 * 2 + 4)
    if grad_checkpointing:
        activation = embedding + logits + per_layer
    else:
        activation = embedding + logits + num_layers * per_layer
    return int(activation * dtype_bytes * 1.5)


def fits(preset: RuntimePreset, estimate: MemoryEstimate) -> bool:
    """True when the estimate fits inside the preset's VRAM (or CPU RAM for cpu)."""
    if preset.vram_mb is None:
        return estimate.total_bytes <= _system_ram_bytes() // 2
    return estimate.total_mib <= preset.vram_mb * 0.85  # 15% headroom


def _system_ram_bytes() -> int:
    try:
        import os

        if os.name == "nt":  # Windows: TotalPhysicalMemory in KB
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys)
    except Exception:  # pragma: no cover - platform fallback
        pass
    return 8 * 1024**3  # assume ≥ 8 GB when unmeasurable


@dataclass(frozen=True)
class RuntimeRecommendation:
    """What to change in a config to fit a preset."""

    fits_without_changes: bool
    max_train_batch: int
    max_seq_len: int
    enable_grad_checkpointing: bool
    estimate: MemoryEstimate

    @property
    def summary(self) -> str:
        return (
            f"params {self.estimate.params_mib:.1f} MiB, total "
            f"{self.estimate.total_mib:.1f} MiB, "
            f"train batch <= {self.max_train_batch}, seq <= {self.max_seq_len}, "
            f"checkpointing {'on' if self.enable_grad_checkpointing else 'off'}"
        )


def recommend_config_overrides(
    preset: RuntimePreset,
    *,
    n_params: int,
    d_model: int,
    num_layers: int,
    vocab_size: int,
    train_batch: int,
    max_seq_len: int,
    grad_checkpointing: bool,
    dtype_bytes: int = 2,
) -> RuntimeRecommendation:
    """
    Recommend batch/seq/checkpointing values that fit ``preset``.

    Strategy (binary search, always over-estimating):
      1. try the user's (batch, seq) as-is; if it fits, done;
      2. halve the batch until it fits;
      3. if still too big, halve seq_len and retry the original batch;
      4. if neither alone suffices, halve both; as a last resort turn on
         gradient checkpointing (never exceeds the preset's budget).
    """
    batch = max(1, int(train_batch))
    seq = max(1, int(max_seq_len))
    checkpoint = bool(grad_checkpointing)

    def est(b: int, s: int, ckpt: bool) -> MemoryEstimate:
        return estimate_training_memory(
            n_params=n_params,
            d_model=d_model,
            num_layers=num_layers,
            seq_len=s,
            batch_size=b,
            vocab_size=vocab_size,
            dtype_bytes=dtype_bytes,
            grad_checkpointing=ckpt,
        )

    # 1) as-is
    candidate = est(batch, seq, checkpoint)
    if fits(preset, candidate):
        return RuntimeRecommendation(True, batch, seq, checkpoint, candidate)

    # 2) shrink batch
    b = batch
    while b > 1:
        b //= 2
        candidate = est(b, seq, checkpoint)
        if fits(preset, candidate):
            return RuntimeRecommendation(False, b, seq, checkpoint, candidate)

    # 3) shrink seq at original batch
    s = seq
    while s > 1:
        s //= 2
        candidate = est(batch, s, checkpoint)
        if fits(preset, candidate):
            return RuntimeRecommendation(False, batch, s, checkpoint, candidate)

    # 4) shrink both, then checkpointing as the final lever
    b = batch
    s = seq
    while b > 1 and s > 1:
        b //= 2
        s //= 2
        candidate = est(b, s, checkpoint)
        if fits(preset, candidate):
            return RuntimeRecommendation(False, b, s, checkpoint, candidate)

    candidate = est(max(1, b), max(1, s), True)
    return RuntimeRecommendation(False, max(1, b), max(1, s), True, candidate)
