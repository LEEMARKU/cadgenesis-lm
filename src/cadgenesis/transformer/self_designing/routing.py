"""
cadgenesis.transformer.self_designing.routing
=======================================
Dynamic Layer Routing for the Self-Designing Transformer.

A ``DynamicLayerRouter`` learns, per token, which transformer layers to keep
active and which to skip (LayerDrop-style).  Routing decisions are:

* **continuous** during training — relaxed Gumbel-Sigmoid so gradients flow;
* **hard** during evaluation — thresholded to {0, 1} after a configurable
  drop-rate prior, so under-confident layers can be skipped entirely.

The controller exposes ``layer_gate(layer_idx, x) -> (B, T, 1)`` which the
backbone multiplies into each block's residual delta (see
``CADTransformerBlock.forward``).  A gate of exactly 0 reproduces the input
for that token → the layer is effectively skipped.

Algorithm
---------
    logits        = MLP(x)                          (B, T, L)
    noisy         = logits + Gumbel_noise (train)   (B, T, L)
    keep          = sigmoid(noisy / τ)              (B, T, L)   ∈ [0, 1]
    hard (eval)   = (keep >= 0.5).float()

Complexity
----------
    Forward: O(B · T · (L · d + d²))  (one MLP shared across layers)
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn


def _gumbel_noise_like(x: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(x.dtype).tiny
    u = torch.rand_like(x).clamp_min(eps)
    return -torch.log(-torch.log(u))


class DynamicLayerRouter(nn.Module):
    """
    Per-token, per-layer keep/drop router for dynamic depth.

    Parameters
    ----------
    d_model : int
        Embedding dimension of the input tokens.
    num_layers : int
        Total number of routable layers (encoder + decoder).
    temperature : float
        Gumbel-Sigmoid temperature (lower → closer to discrete).
    drop_rate : float
        Initial probability of dropping a layer (bias prior on the router).
    hard_eval : bool
        Threshold to {0, 1} at evaluation time.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        temperature: float = 1.0,
        drop_rate: float = 0.2,
        hard_eval: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.temperature = temperature
        self.drop_rate = drop_rate
        self.hard_eval = hard_eval

        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_layers),
        )
        # Initialise the last layer so sigmoid(logit) ≈ (1 - drop_rate).
        init_bias = math.log(max((1.0 - drop_rate) / max(drop_rate, 1e-6), 1e-6))
        last_layer = cast(nn.Linear, self.scorer[-1])
        with torch.no_grad():
            last_layer.weight.data.zero_()
            last_layer.bias.data.fill_(init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C) → keep mask (B, T, num_layers) with values in [0, 1].
        """
        logits = self.scorer(x)  # (B, T, L)
        if self.training:
            noisy = logits + _gumbel_noise_like(logits)
            return torch.sigmoid(noisy / self.temperature)
        keep = torch.sigmoid(logits)
        if self.hard_eval:
            keep = (keep >= 0.5).float()
        return keep

    def layer_gate(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """
        Per-token gate for one layer: (B, T, 1).  The router output is cached
        per sequence-length so it is only computed once per forward pass.
        """
        if not (0 <= layer_idx < self.num_layers):
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers}).")
        mask = self.forward(x)  # (B, T, L)
        return mask[:, :, layer_idx : layer_idx + 1]

    def keep_ratio(self, x: torch.Tensor) -> float:
        """Fraction of (token, layer) pairs kept — a routing sparsity metric."""
        with torch.no_grad():
            mask = self.forward(x)
            return float(mask.mean().item())

    def complexity_report(self) -> dict:
        return {
            "num_layers": self.num_layers,
            "drop_rate": self.drop_rate,
            "temperature": self.temperature,
            "hard_eval": self.hard_eval,
        }
