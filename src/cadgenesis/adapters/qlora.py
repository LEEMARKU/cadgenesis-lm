"""cadgenesis.adapters.qlora
==========================
QLoRA (quantized LoRA) implementation.

Pure-torch implementation (no bitsandbytes): base Linear weights are
quantized to int8 with per-channel absmax symmetric quantization
(``scale = max|w| / 127``, ``w_int8 = round(w / scale)``, one scale per
out-channel) and dequantized on the fly during forward. LoRA is then attached
by reusing :class:`cadgenesis.adapters.lora.LoRALinear`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cadgenesis.adapters.lora import LoRALinear

_INT8_MAX = 127.0
DEFAULT_TARGET_LAYERS = ["q_proj", "k_proj", "v_proj", "out_proj"]


class QuantizedLinear(nn.Module):
    # type: ignore[file]
    """int8 per-channel quantized linear (dequantizes on forward)."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        weight = linear.weight.detach()
        scale = (weight.abs().amax(dim=1).clamp(min=1e-12)) / _INT8_MAX
        quantized = torch.round(weight / scale[:, None]).clamp(-_INT8_MAX, _INT8_MAX)
        self.scale: torch.Tensor
        self.weight_int8: torch.Tensor
        self.register_buffer("scale", scale)
        self.register_buffer("weight_int8", quantized.to(torch.int8))
        self.bias = linear.bias.detach().clone() if linear.bias is not None else None
        self.original_bytes = weight.numel() * 4 + (
            linear.bias.numel() * 4 if linear.bias is not None else 0
        )

    @property
    def weight(self) -> torch.Tensor:
        """Dequantized weights (int8 * per-channel scale)."""
        return self.weight_int8.float() * self.scale[:, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def quantized_bytes(self) -> int:
        """Bytes after quantization (int8 weights + fp32 scale and bias)."""
        total = int(self.weight_int8.numel()) + int(self.scale.numel()) * 4
        if self.bias is not None:
            total += self.bias.numel() * 4
        return total


class QuantizedModel(nn.Module):
    """Module wrapping a model whose Linear layers are QuantizedLinear."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.original_bytes_total = 0
        self.quantized_bytes_total = 0
        self._quantize_module(model)

    def _quantize_module(self, module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                quantized = QuantizedLinear(child)
                self.original_bytes_total += quantized.original_bytes
                self.quantized_bytes_total += quantized.quantized_bytes()
                setattr(module, name, quantized)
            else:
                self._quantize_module(child)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def memory_report(self) -> dict[str, float]:
        """original / quantized byte counts and percent savings (pre-LoRA)."""
        savings_pct = 0.0
        if self.original_bytes_total > 0:
            savings_pct = (1.0 - self.quantized_bytes_total / self.original_bytes_total) * 100.0
        return {
            "original_bytes": float(self.original_bytes_total),
            "quantized_bytes": float(self.quantized_bytes_total),
            "savings_pct": savings_pct,
        }


class QLoRAAdapter:
    """Quantizes a model to int8 and attaches LoRA on top of it."""

    def __init__(self, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05) -> None:
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self._attached: str | None = None

    @property
    def attached(self) -> str | None:
        return self._attached

    def quantize(self, model: nn.Module) -> QuantizedModel:
        """Return a :class:`QuantizedModel` wrapping ``model`` (mutates in place)."""
        return QuantizedModel(model)

    def attach_lora(
        self,
        qmodel: QuantizedModel,
        adapter_id: str,
        target_layers: list[str] | None = None,
    ) -> None:
        """Attach LoRA (reusing lora.LoRALinear) onto QuantizedLinear layers."""
        if self._attached is not None:
            raise ValueError(f"LoRA adapter {self._attached!r} is already attached")
        targets = target_layers if target_layers is not None else DEFAULT_TARGET_LAYERS
        installed = 0
        for _name, module in qmodel.named_modules():
            for attr_name, child in module.named_children():
                if isinstance(child, QuantizedLinear) and any(
                    target in attr_name for target in targets
                ):
                    setattr(
                        module,
                        attr_name,
                        LoRALinear(child, rank=self.rank, alpha=self.alpha, dropout=self.dropout),
                    )
                    installed += 1
        if installed == 0:
            raise ValueError(f"no QuantizedLinear layers matched target_layers={targets!r}")
        self._attached = adapter_id

    def memory_report(self, model: QuantizedModel) -> dict[str, float]:
        """Memory footprint of the quantized model."""
        return model.memory_report()

    def merge(self, qmodel: QuantizedModel) -> nn.Module:
        """Fold quantization + LoRA delta back into plain nn.Linear modules."""
        self._fold_linears(qmodel.model)
        self._attached = None
        return qmodel.model

    def _fold_linears(self, module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear) and isinstance(child.original_linear, QuantizedLinear):
                quantized = child.original_linear
                delta = (child.lora_B @ child.lora_A) * child.scaling
                self._replace_with_linear(module, name, quantized, delta)
            elif isinstance(child, QuantizedLinear):
                self._replace_with_linear(module, name, child, delta=None)
            else:
                self._fold_linears(child)

    @staticmethod
    def _replace_with_linear(
        module: nn.Module,
        name: str,
        quantized: QuantizedLinear,
        delta: torch.Tensor | None,
    ) -> None:
        linear = nn.Linear(
            quantized.in_features, quantized.out_features, bias=quantized.bias is not None
        )
        with torch.no_grad():
            weight = quantized.weight
            if delta is not None:
                weight = weight + delta
            linear.weight.copy_(weight)
            if quantized.bias is not None:
                linear.bias.copy_(quantized.bias)
        setattr(module, name, linear)
