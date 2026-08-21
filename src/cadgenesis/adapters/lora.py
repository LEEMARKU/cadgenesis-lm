r"""
cadgenesis.adapters.lora
========================
Parameter-Efficient Fine-Tuning (PEFT), LoRA, and QLoRA for CADGenesis-LM v2.0.

Provides:
- LoRALayer: Low-Rank Adaptation linear layer wrapper ($W = W_0 + \frac{\alpha}{r} B \cdot A$)
- apply_lora_to_model: Helper function injecting LoRA adapters into targeted linear modules
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA Linear Layer Wrapper:
    Wraps an existing nn.Linear layer and computes low-rank updates.
    """

    def __init__(
        self, original_linear: Any, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05
    ):
        super().__init__()
        self.original_linear: Any = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_dim = original_linear.in_features
        out_dim = original_linear.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        self.dropout = nn.Dropout(dropout)

        # Freeze original linear weights
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        # Initialize A with Gaussian and B with Zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original_linear(x)
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return base_out + (lora_out * self.scaling)


def apply_lora(
    model: nn.Module, rank: int = 16, alpha: float = 32.0, target_modules: list[str] | None = None
):
    """
    Injects LoRALinear layers into all targeted linear modules in model.
    """
    target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "out_proj"]
    for _name, module in model.named_modules():
        for attr_name, child in module.named_children():
            if isinstance(child, nn.Linear) and any(t in attr_name for t in target_modules):
                setattr(module, attr_name, LoRALinear(child, rank=rank, alpha=alpha))
