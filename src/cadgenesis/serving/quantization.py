"""
cadgenesis.serving.quantization
===============================
Model quantization for deployment (pure torch, no external deps).

Provides two inference-oriented quantized linear layers:

* ``FP8Linear`` — per-output-channel int8 weight + fp32 scale (FP8-style).
* ``INT4Linear`` — group-wise signed INT4 weight with RTN rounding, stored
  as int8 values in [-7, 7] with a per-group fp32 scale.

plus ``quantize_model`` (recursive in-place ``nn.Linear`` replacement) and
``report_quantization`` (memory-savings report).

Both layers are designed for *inference*: weights are integer buffers that
are dequantized on the fly in ``forward``.  Gradients flow through the fp32
scale/bias parameters (so ``backward`` does not crash), but the integer
weights themselves are not trainable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "FP8Linear",
    "INT4Linear",
    "quantize_model",
    "report_quantization",
]


class FP8Linear(nn.Module):
    """FP8-style linear layer: per-output-channel int8 weight + fp32 scale.

    Wraps a ``nn.Linear``: the weight is stored as an int8 tensor
    (``weight_int8``, shape (out_features, in_features)) plus a float32
    per-output-channel ``scale``; the original fp32 ``bias`` (or None) is
    kept unchanged.  ``forward`` dequantizes on the fly:
    ``w = weight_int8.float() * scale.view(-1, 1)`` then ``F.linear``.
    """

    weight_int8: torch.Tensor
    scale: nn.Parameter
    bias: nn.Parameter | None

    def __init__(
        self,
        weight_int8: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        if weight_int8.dtype != torch.int8:
            raise ValueError(f"weight_int8 must be an int8 tensor, got {weight_int8.dtype}")
        if scale.ndim != 1 or scale.shape[0] != weight_int8.shape[0]:
            raise ValueError(
                "scale must be 1-D with one value per output channel "
                f"(expected {weight_int8.shape[0]}, got {tuple(scale.shape)})"
            )
        self.register_buffer("weight_int8", weight_int8)
        self.scale = nn.Parameter(scale)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)

    @property
    def in_features(self) -> int:
        return self.weight_int8.shape[1]

    @property
    def out_features(self) -> int:
        return self.weight_int8.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize the weight and apply the linear transform."""
        w = self.weight_int8.float() * self.scale.view(-1, 1)
        return F.linear(x, w, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> FP8Linear:
        """Quantize a ``nn.Linear`` per output channel: scale = max|W_c| / 127."""
        w = linear.weight.detach()
        scale = w.abs().amax(dim=1) / 127.0  # (out,)
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        weight_int8 = torch.round(w / safe.view(-1, 1)).clamp(-127, 127).to(torch.int8)
        bias = linear.bias.detach() if linear.bias is not None else None
        return cls(weight_int8, scale, bias)

    def memory_bytes(self) -> int:
        """Stored bytes: int8 weight + fp32 scale + fp32 bias."""
        bias_bytes = self.out_features * 4 if self.bias is not None else 0
        return self.weight_int8.numel() * 1 + self.out_features * 4 + bias_bytes

    def fp32_memory_bytes(self) -> int:
        """Bytes of the equivalent fp32 weight + bias (for savings reporting)."""
        bias_bytes = self.out_features * 4 if self.bias is not None else 0
        return self.weight_int8.numel() * 4 + bias_bytes


class INT4Linear(nn.Module):
    """Group-wise signed INT4 linear layer with RTN (round-to-nearest) rounding.

    Columns (``in_features``) are split into groups of ``group_size``; every
    group has its own fp32 scale ``max(|W_g|) / 7`` and its weights are
    stored as int8 values in [-7, 7].  ``forward`` dequantizes per group:
    ``w = q.float() * scale`` reshaped to (out_features, in_features), then
    ``F.linear``.

    ``in_features`` must be divisible by ``group_size``.  If it is not, a
    ``ValueError`` is raised and the caller should pick a ``group_size`` that
    divides ``in_features`` (rather than padding the remainder group).
    """

    weight_int8: torch.Tensor
    scale: nn.Parameter
    bias: nn.Parameter | None
    in_features: int
    out_features: int
    group_size: int
    num_groups: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by group_size "
                f"({group_size}); choose a group_size that divides in_features"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.num_groups = in_features // group_size
        self.register_buffer(
            "weight_int8",
            torch.zeros(out_features, in_features, dtype=torch.int8),
        )
        self.scale = nn.Parameter(torch.ones(out_features, self.num_groups))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize group-wise and apply the linear transform."""
        out, groups, gs = self.out_features, self.num_groups, self.group_size
        w = (self.weight_int8.float().view(out, groups, gs) * self.scale.unsqueeze(-1)).view(
            out, self.in_features
        )
        return F.linear(x, w, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 128) -> INT4Linear:
        """Quantize a ``nn.Linear`` group-wise: scale = max|W_g| / 7, RTN to [-7, 7]."""
        w = linear.weight.detach()
        out, inn = w.shape
        if inn % group_size != 0:
            raise ValueError(
                f"in_features ({inn}) must be divisible by group_size "
                f"({group_size}); choose a group_size that divides in_features"
            )
        num_groups = inn // group_size
        wg = w.view(out, num_groups, group_size)
        scale = wg.abs().amax(dim=-1) / 7.0  # (out, num_groups)
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        q = torch.round(wg / safe.unsqueeze(-1)).clamp(-7, 7).to(torch.int8)

        mod = cls(inn, out, group_size=group_size, bias=linear.bias is not None)
        mod.weight_int8.copy_(q.view(out, inn))
        mod.scale.data.copy_(scale)
        if mod.bias is not None and linear.bias is not None:
            mod.bias.data.copy_(linear.bias.detach())
        return mod

    def memory_bytes(self) -> int:
        """Stored bytes: int8 weight + fp32 per-group scales + fp32 bias."""
        bias_bytes = self.out_features * 4 if self.bias is not None else 0
        return self.weight_int8.numel() * 1 + self.scale.numel() * 4 + bias_bytes

    def fp32_memory_bytes(self) -> int:
        """Bytes of the equivalent fp32 weight + bias (for savings reporting)."""
        bias_bytes = self.out_features * 4 if self.bias is not None else 0
        return self.weight_int8.numel() * 4 + bias_bytes


def quantize_model(
    model: nn.Module,
    mode: str = "fp8",
    group_size: int = 128,
    skip_classes: tuple = (nn.Embedding, nn.LayerNorm),
) -> nn.Module:
    """Recursively replace every ``nn.Linear`` with a quantized equivalent.

    Parameters
    ----------
    model : nn.Module
        Model to quantize (modified in place and also returned).
    mode : str
        ``"fp8"`` → ``FP8Linear``, ``"int4"`` → ``INT4Linear``,
        ``"none"`` → returned unchanged.
    group_size : int
        Group size for INT4 quantization.
    skip_classes : tuple
        Module classes left untouched (e.g. ``nn.Embedding``, ``nn.LayerNorm``).

    Returns
    -------
    nn.Module
        The (mutated) model with quantized linear layers.
    """
    if mode not in ("fp8", "int4", "none"):
        raise ValueError(f"unknown quantization mode {mode!r}; expected 'fp8', 'int4', or 'none'")
    if mode == "none":
        return model

    def wrap(linear: nn.Linear) -> nn.Module:
        if mode == "fp8":
            return FP8Linear.from_linear(linear)
        return INT4Linear.from_linear(linear, group_size=group_size)

    if isinstance(model, nn.Linear) and not isinstance(model, skip_classes):
        return wrap(model)

    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, skip_classes):
            setattr(model, name, wrap(child))
        else:
            quantize_model(child, mode=mode, group_size=group_size, skip_classes=skip_classes)
    return model


def report_quantization(model: nn.Module, fp32_model: nn.Module) -> dict:
    """Report memory before/after quantization.

    ``fp32_bytes`` counts the fp32 weight+bias bytes of every linear layer in
    ``fp32_model``; ``quant_bytes`` counts the stored bytes of the quantized
    layers in ``model`` (any remaining unquantized ``nn.Linear`` is counted at
    fp32 size).  ``ratio = quant_bytes / fp32_bytes``.

    Returns
    -------
    dict
        ``{"fp32_bytes": int, "quant_bytes": int, "ratio": float, "n_linears": int}``
    """
    fp32_bytes = 0
    n_linears = 0
    for m in fp32_model.modules():
        if isinstance(m, nn.Linear):
            n_linears += 1
            if isinstance(m, (FP8Linear, INT4Linear)):
                fp32_bytes += int(m.fp32_memory_bytes())
            else:
                fp32_bytes += m.weight.numel() * 4
                if m.bias is not None:
                    fp32_bytes += m.bias.numel() * 4

    quant_bytes = 0
    for m in model.modules():
        if isinstance(m, (FP8Linear, INT4Linear)):
            quant_bytes += int(m.memory_bytes())
        elif isinstance(m, nn.Linear):
            quant_bytes += m.weight.numel() * 4
            if m.bias is not None:
                quant_bytes += m.bias.numel() * 4

    return {
        "fp32_bytes": int(fp32_bytes),
        "quant_bytes": int(quant_bytes),
        "ratio": float(quant_bytes / fp32_bytes) if fp32_bytes else 0.0,
        "n_linears": int(n_linears),
    }
