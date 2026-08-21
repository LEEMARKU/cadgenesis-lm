"""
cadgenesis.transformer.multi_scale_attention
============================================
Multi-Scale Attention for CADGenesis-LM v6.0 (Pillar 1).

CAD sequences encode structure at several granularities at once: local sketch
geometry (a few neighbouring tokens), medium-range feature chains (a pad →
hole → chamfer sequence), and global assembly/simulation context.  Plain
single-window attention forces the model to pick one receptive field.
:class:`MultiScaleAttention` runs **local**, **medium** and **global**
attention heads *simultaneously* and concatenates their outputs, so every token
aggregates information across all three scales in a single layer.

Head budget
-----------
``num_heads`` is split across the three scales according to ``head_fractions``
(a tuple summing to 1.0, default ``(0.5, 0.3, 0.2)``).  Each scale head uses the
corresponding band width (``local_window`` / ``medium_window``); the global
group uses the full (causal) range.

Complexity
----------
    Local/medium heads:  O(T · max(local_window, medium_window) · C)
    Global heads:        O(T² / 5)  (only ~20% of heads are quadratic)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cadgenesis.transformer.positional import RotaryEmbedding
from cadgenesis.transformer.sparse_attention import sparse_attention_mask


class MultiScaleAttention(nn.Module):
    """
    Parallel local + medium + global multi-head attention.

    Parameters
    ----------
    d_model : int
        Model embedding dimension.
    num_heads : int
        Total number of heads (split across the three scales).
    dropout : float
        Attention dropout.
    head_fractions : tuple[float, float, float]
        ``(local, medium, global)`` fractions summing to 1.0.
    local_window : int
        Band width of the local scale.
    medium_window : int
        Band width of the medium scale.
    causal : bool
        Restrict every scale to ``j <= i`` (autoregressive decoding).
    use_sdpa : bool
        Use the fused SDPA kernel when available.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        head_fractions: tuple[float, float, float] = (0.5, 0.3, 0.2),
        local_window: int = 64,
        medium_window: int = 256,
        causal: bool = True,
        use_sdpa: bool = False,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if len(head_fractions) != 3:
            raise ValueError("head_fractions must be a 3-tuple (local, medium, global).")
        if abs(sum(head_fractions) - 1.0) > 1e-6:
            raise ValueError("head_fractions must sum to 1.0.")
        if any(f < 0 for f in head_fractions):
            raise ValueError("head_fractions must be non-negative.")
        if local_window < 1 or medium_window < local_window:
            raise ValueError("require 1 <= local_window <= medium_window.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.local_window = local_window
        self.medium_window = medium_window
        self.causal = causal
        self.use_sdpa = use_sdpa
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

        # Scale head counts (round the local/medium groups, remainder → global).
        local_h = round(num_heads * head_fractions[0])
        medium_h = round(num_heads * head_fractions[1])
        global_h = num_heads - local_h - medium_h
        if min(local_h, medium_h, global_h) < 1:
            raise ValueError(
                "head_fractions must leave every scale with >= 1 head "
                f"(local={local_h}, medium={medium_h}, global={global_h} for "
                f"num_heads={num_heads})."
            )
        self.local_heads = local_h
        self.medium_heads = medium_h
        self.global_heads = global_h
        self._scale_offsets = (
            0,
            local_h,
            local_h + medium_h,
            local_h + medium_h + global_h,
        )

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self._mask_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def _masks_for(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (local, medium) additive masks, cached per seq length."""
        key = (seq_len, str(device))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        local = sparse_attention_mask(
            "local",
            seq_len,
            window_size=self.local_window,
            num_global_tokens=1,
            block_size=1,
            causal=self.causal,
            device=device,
        )
        medium = sparse_attention_mask(
            "local",
            seq_len,
            window_size=self.medium_window,
            num_global_tokens=1,
            block_size=1,
            causal=self.causal,
            device=device,
        )
        self._mask_cache[key] = (local, medium)
        return local, medium

    def _group(self, tensor: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Slice head-group ``[start, end)`` out of a (B, H, T, D) tensor."""
        return tensor[:, start:end]

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        use_rope: bool = True,
    ) -> torch.Tensor:
        """
        x: (B, T, C)
        attn_mask: optional additive mask (1, 1, T, T) applied to all scales.
        Returns (B, T, C).
        """
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if use_rope:
            q, k = self.rope(q, k)

        local_mask, medium_mask = self._masks_for(T, x.device)

        def attend(qg, kg, vg, mask):
            if self.use_sdpa:
                if attn_mask is not None:
                    mask = mask + attn_mask if mask is not None else attn_mask
                return F.scaled_dot_product_attention(
                    qg,
                    kg,
                    vg,
                    attn_mask=mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=False,
                )
            scores = torch.matmul(qg, kg.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            if attn_mask is not None:
                scores = scores + attn_mask
            probs = F.softmax(scores, dim=-1)
            probs = self.dropout(probs)
            return torch.matmul(probs, vg)

        o0 = self._scale_offsets[0]
        o1, o2, o3 = self._scale_offsets[1], self._scale_offsets[2], self._scale_offsets[3]

        local_out = attend(
            self._group(q, o0, o1), self._group(k, o0, o1), self._group(v, o0, o1), local_mask
        )
        medium_out = attend(
            self._group(q, o1, o2), self._group(k, o1, o2), self._group(v, o1, o2), medium_mask
        )
        global_out = attend(
            self._group(q, o2, o3), self._group(k, o2, o3), self._group(v, o2, o3), None
        )

        out = torch.cat([local_out, medium_out, global_out], dim=1)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))

    @property
    def scale_report(self) -> dict[str, int]:
        """Per-scale head counts (diagnostics)."""
        return {
            "local": self.local_heads,
            "medium": self.medium_heads,
            "global": self.global_heads,
            "local_window": self.local_window,
            "medium_window": self.medium_window,
        }
