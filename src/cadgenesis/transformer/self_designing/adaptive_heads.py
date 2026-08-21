"""
cadgenesis.transformer.self_designing.adaptive_heads
==============================================
Adaptive Attention Heads for the Self-Designing Transformer.

The ``MultiHeadAttentionMixture`` in every block already combines six
specialised head types (Self, Geometry, Constraint, Memory, Agent,
Uncertainty) with a learned softmax gate.  ``AdaptiveAttentionHeadSelector``
learns an *additional, per-token* modulation over those head types so the
model can dynamically emphasise or switch off whole head classes:

* soft & differentiable during training (relaxed Gumbel-Sigmoid);
* thresholded to {0, 1} at evaluation for cheaper inference.

The selector is shared across all blocks (one scorer of width
``num_layers * num_active_heads``), matching the backbone's duck-typed
``head_weights(block_idx, x)`` interface.

Algorithm
---------
    logits = Linear(x)                  (B, T, L·H)
    keep   = gumbel_sigmoid(logits / τ) (B, T, L·H)
    hw     = keep[:, :, l·H:(l+1)·H]    (B, T, H)

Complexity
----------
    Forward: O(B · T · L · H · d)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cadgenesis.transformer.self_designing.routing import _gumbel_noise_like


class AdaptiveAttentionHeadSelector(nn.Module):
    """
    Per-token adaptive gating over the attention mixture's active head types.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    num_layers : int
        Total routable layers (encoder + decoder).
    num_active_heads : int
        Number of active head types in ``MultiHeadAttentionMixture``
        (i.e. ``mixture.num_active``).  Must match for every block.
    drop_rate : float
        Initial probability of zeroing a head type.
    temperature : float
        Gumbel-Sigmoid temperature.
    hard_eval : bool
        Threshold to {0, 1} at evaluation.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_active_heads: int,
        drop_rate: float = 0.3,
        temperature: float = 1.0,
        hard_eval: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_active_heads = num_active_heads
        self.temperature = temperature
        self.drop_rate = drop_rate
        self.hard_eval = hard_eval

        self.scorer = nn.Linear(d_model, num_layers * num_active_heads)
        with torch.no_grad():
            self.scorer.weight.data.normal_(0.0, 0.02)
            init_bias = torch.log(torch.tensor((1.0 - drop_rate) / max(drop_rate, 1e-6)))
            self.scorer.bias.data.fill_(float(init_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C) → per-layer head keep masks (B, T, num_layers, H).
        """
        B, T, _ = x.shape
        logits = self.scorer(x).view(B, T, self.num_layers, self.num_active_heads)
        if self.training:
            noisy = logits + _gumbel_noise_like(logits)
            return torch.sigmoid(noisy / self.temperature)
        keep = torch.sigmoid(logits)
        if self.hard_eval:
            keep = (keep >= 0.5).float()
        return keep

    def head_weights(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """(B, T, H) modulation for a single layer."""
        if not (0 <= layer_idx < self.num_layers):
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers}).")
        masks = self.forward(x)
        return masks[:, :, layer_idx, :]

    def active_head_ratio(self, x: torch.Tensor) -> float:
        """Fraction of (token, head) pairs kept — a head-sparsity metric."""
        with torch.no_grad():
            masks = self.forward(x)
            return float(masks.mean().item())
