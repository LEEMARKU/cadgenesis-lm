"""
cadgenesis.training.optimizer
=============================
Optimizer factory for CADGenesis-LM training.

``build_optimizer`` constructs AdamW/Adam with weight decay and optional
LoRA-only parameter filtering (only trainable parameters are tuned; frozen
base weights are excluded automatically).  ``lora_param_groups`` builds the
parameter groups used by LoRA fine-tuning.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    import torch
except ImportError:  # pragma: no cover - torch optional
    torch = None  # type: ignore[assignment]

OPTIMIZERS = ("adamw", "adamw8bit", "adam", "sgd", "lamb")


def _adamw8bit(params, lr, betas, weight_decay):
    """8-bit AdamW (bitsandbytes) — 1.5B-scale models on limited VRAM/RAM."""
    try:
        from bitsandbytes.optim import AdamW8bit
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("adamw8bit requires `pip install bitsandbytes`") from exc
    return AdamW8bit(params, lr=lr, betas=betas, weight_decay=weight_decay)


def lora_param_groups(
    model: Any,
    lr: float = 3e-4,
    base_lr: float | None = None,
    weight_decay: float = 0.0,
    lora_modules: tuple[str, ...] = ("lora",),
) -> list[dict[str, Any]]:
    """Two parameter groups: LoRA params (full LR) and other trainable params.

    Frozen parameters are excluded entirely, so training only touches the
    adapter parameters plus any other trainable weights.
    """
    lora_params: list[torch.nn.Parameter] = []
    other_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = lora_params if any(tag in name for tag in lora_modules) else other_params
        group.append(param)
    groups: list[dict[str, Any]] = []
    if lora_params:
        groups.append({"params": lora_params, "lr": lr, "weight_decay": weight_decay})
    if other_params and base_lr is not None:
        groups.append({"params": other_params, "lr": base_lr, "weight_decay": weight_decay})
    elif other_params:
        groups.append({"params": other_params, "lr": lr, "weight_decay": weight_decay})
    if not groups:
        raise ValueError("no trainable parameters found")
    return groups


def build_optimizer(
    model: Any,
    optimizer_type: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    momentum: float = 0.9,
    lora_only: bool = False,
) -> Any:
    """Build an optimizer over ``model``'s trainable parameters.

    ``lora_only=True`` restricts training to LoRA adapter parameters
    (requires at least one trainable LoRA parameter).
    """
    if torch is None:  # pragma: no cover - torch optional
        raise ImportError("build_optimizer requires torch")
    if optimizer_type not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {optimizer_type!r}; expected one of {OPTIMIZERS}")
    if lora_only:
        params = lora_param_groups(model, lr=lr, weight_decay=weight_decay)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("model has no trainable parameters")
    if optimizer_type == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if optimizer_type == "adamw8bit":
        return _adamw8bit(params, lr, betas, weight_decay)
    if optimizer_type == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if optimizer_type == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)


__all__ = ["OPTIMIZERS", "build_optimizer", "lora_param_groups"]
