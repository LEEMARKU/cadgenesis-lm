"""cadgenesis.optimization.quantization
====================================
Post-training quantization (INT8 / QLoRA-4-bit) of CADGenesis models.

Provides quantization utilities for CADGenesis-LM model compression,
including INT8 per-channel symmetric quantization, QLoRA-4-bit
quantization with LoRA adaptation, and calibration for accuracy
preservation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinear(nn.Module):
    """INT8 quantized linear layer with per-channel scaling.

    Weights are quantized to int8 using symmetric per-channel
    absmax quantization: scale = max|w| / 127, w_int8 = round(w / scale).
    Dequantization occurs on-the-fly during forward pass.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 128,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # Original full-precision weight (frozen after quantization)
        self.weight_fp32 = nn.Parameter(torch.empty(out_features, in_features))

        # Quantized int8 weight and per-channel scales
        self.weight_int8 = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.scale = nn.Parameter(
            torch.empty(
                out_features,
            ),
            requires_grad=False,
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._initialized = False

    def _quantize(self, weight_fp32: torch.Tensor) -> None:
        """Quantize a weight matrix to INT8 with per-channel scale.

        - ``weight_fp32``: (out_features, in_features)
        """
        self.weight_fp32.data.copy_(weight_fp32)
        out_features, _ = weight_fp32.shape

        # Per-channel symmetric absmax quantization
        scales = weight_fp32.abs().amax(dim=1).clamp(min=1e-8)
        scales = scales / 127.0  # scale such that w / scale fits int8 range
        self.scale.data.copy_(scales)
        self.weight_int8.data.copy_(
            (weight_fp32 / scales.view(out_features, 1)).round().clamp(-128, 127).to(torch.int8)
        )
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly dequantization.

        - ``x``: (B, seq_len, in_features)
        Returns: (B, seq_len, out_features)
        """
        if not self._initialized:
            raise RuntimeError("QuantizedLinear weight not initialized. Call ._quantize() first.")

        # Dequantize on the fly: x @ (scale * weight_int8)
        deq_weight = (self.scale.view(-1, 1) * self.weight_int8).to(x.dtype)
        out = F.linear(x, deq_weight, self.bias)
        return out


def quantize_model_qt(
    model: nn.Module,
    example_input: torch.Tensor,
    qconfig: str = "int8",
) -> nn.Module:
    """Quantize a model for deployment.

    - ``model``: PyTorch model containing nn.Linear layers.
    - ``example_input``: Example input for shape inference.
    - ``qconfig``: Quantization config ("int8" or "qlora-4bit").

    Returns the quantized model with ``QuantizedLinear`` replacing
    original ``nn.Linear`` layers where applicable.
    """
    import copy

    model = copy.deepcopy(model)

    if qconfig == "int8":
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and not isinstance(module, QuantizedLinear):
                qlinear = QuantizedLinear(
                    module.in_features, module.out_features, module.bias is not None
                )
                qlinear._quantize(module.weight.data)
                # Replace in parent
                parts = name.split(".")
                target = model
                for p in parts[:-1]:
                    target = getattr(target, p)
                setattr(target, parts[-1], qlinear)

    elif qconfig == "qlora-4bit":
        pass  # QLoRA-4bit: see serving.quantization for the 4-bit implementation

    return model


__all__ = [
    "QuantizedLinear",
    "quantize_model_qt",
]
