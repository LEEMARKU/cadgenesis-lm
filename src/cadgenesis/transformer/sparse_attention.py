"""
cadgenesis.transformer.sparse_attention
=======================================
Scalable Sparse Attention for CADGenesis-LM v6.0 (Pillar 1).

Sparse attention replaces the dense ``T x T`` score matrix with a structured
sparse pattern so long CAD sequences (high-resolution geometry sketches,
assembly graphs, simulation meshes) can be processed in sub-quadratic time and
memory.  Five patterns are provided:

1. ``local``           — each query attends to its immediate ``window_size``
                         predecessors (O(T · W), W << T).
2. ``global``          — BigBird-style: a small set of *global* tokens attend to
                         everything and everything attends back to them, combined
                         with the causal diagonal (O(T · G + T), G = #globals).
3. ``sliding_window``  — Mistral-style banded attention: query ``i`` attends to
                         keys ``[i - W + 1, i]`` (O(T · W)).
4. ``block_sparse``    — full attention within fixed blocks of size
                         ``block_size`` plus global tokens (O(T · B + T · G)).
5. ``mixed``           — union of the sliding-window band and global tokens.

All patterns are *causal-capable*: the default mask is upper-triangular so the
module can be dropped into autoregressive decoding unchanged.

A pure function :func:`sparse_attention_mask` exposes the additive
``(1, 1, T, T)`` bias matrix (``0.0`` allowed, ``-inf`` masked) so researchers
can inspect or reuse the patterns directly, and
:func:`build_sparse_attention` mirrors ``build_self_attention`` from
:mod:`cadgenesis.transformer.efficient_attention`.

Complexity
----------
    Memory:  O(T · max(W, G, B))  versus  O(T²)  for dense attention.
    Time:    O(T · max(W, G, B) · C)  (sparse) versus O(T² · C).
"""

from __future__ import annotations

import logging
import math
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from cadgenesis.transformer.positional import RotaryEmbedding

logger = logging.getLogger(__name__)


class SparseAttentionPattern(str, Enum):
    """Supported sparse attention patterns."""

    LOCAL = "local"
    GLOBAL = "global"
    SLIDING_WINDOW = "sliding_window"
    BLOCK_SPARSE = "block_sparse"
    MIXED = "mixed"


SPARSE_PATTERNS = tuple(p.value for p in SparseAttentionPattern)


def sparse_attention_mask(
    pattern: str,
    seq_len: int,
    *,
    window_size: int = 128,
    num_global_tokens: int = 32,
    block_size: int = 64,
    causal: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Build the additive sparse attention mask of shape ``(1, 1, seq_len, seq_len)``.

    ``0.0`` marks an allowed attention pair and ``-inf`` a masked pair.  When
    ``causal=True`` the mask is intersected with ``j <= i`` so future tokens are
    never visible (suitable for the decoder).

    Parameters
    ----------
    pattern : str
        One of ``"local" | "global" | "sliding_window" | "block_sparse" | "mixed"``.
    seq_len : int
        Sequence length.
    window_size : int
        Band width for ``local`` / ``sliding_window`` / ``mixed``.
    num_global_tokens : int
        Number of leading *global* tokens for ``global`` / ``block_sparse`` /
        ``mixed``.
    block_size : int
        Block width for ``block_sparse``.
    causal : bool
        Restrict to the lower triangle ``j <= i``.
    device, dtype :
        Output tensor placement.

    Returns
    -------
    An additive mask ``(1, 1, T, T)``.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if num_global_tokens < 1:
        raise ValueError("num_global_tokens must be >= 1")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if pattern not in SPARSE_PATTERNS:
        raise ValueError(f"Unknown sparse pattern {pattern!r}; choose from {SPARSE_PATTERNS}.")

    i = torch.arange(seq_len, device=device).unsqueeze(1)  # (T, 1) queries
    j = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, T) keys
    rel = i - j  # positive when key precedes query

    if pattern in (SparseAttentionPattern.LOCAL.value, SparseAttentionPattern.SLIDING_WINDOW.value):
        allowed = (rel >= 0) & (rel < window_size)
    elif pattern == SparseAttentionPattern.GLOBAL.value:
        is_global_q = i < num_global_tokens
        is_global_k = j < num_global_tokens
        allowed = is_global_q | is_global_k | (rel >= 0)
    elif pattern == SparseAttentionPattern.BLOCK_SPARSE.value:
        same_block = (i // block_size) == (j // block_size)
        is_global_k = j < num_global_tokens
        allowed = same_block | is_global_k
    elif pattern == SparseAttentionPattern.MIXED.value:
        band = (rel >= 0) & (rel < window_size)
        is_global_k = j < num_global_tokens
        allowed = band | is_global_k
    else:  # pragma: no cover - guarded above
        raise AssertionError(f"Unreachable pattern {pattern!r}")

    if causal:
        allowed = allowed & (rel >= 0)

    float_mask = torch.zeros(seq_len, seq_len, device=device)
    float_mask[~allowed] = float("-inf")
    if dtype is not None:
        float_mask = float_mask.to(dtype)
    return float_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)


def pattern_complexity(pattern: str, seq_len: int, **kwargs: int) -> str:
    """Human-readable complexity summary for a pattern at a given sequence length."""
    w = kwargs.get("window_size", kwargs.get("sliding_window_size", 128))
    g = kwargs.get("num_global_tokens", 32)
    b = kwargs.get("block_size", 64)
    scale = max(w, g, b) if pattern != "block_sparse" else max(b, g)
    dense = seq_len * seq_len
    sparse = seq_len * min(scale, seq_len)
    return (
        f"{pattern}: O(T·{min(scale, seq_len)}) ≈ {sparse:,} pairs "
        f"(dense would be {dense:,}, {dense / max(sparse, 1):.1f}x larger)"
    )


class SparseSelfAttention(nn.Module):
    """
    Multi-head self-attention restricted to a configurable sparse pattern.

    Identical I/O contract to :class:`cadgenesis.transformer.attention.SelfAttention`
    (``forward(x, attn_mask=None, use_rope=True) -> (B, T, d_model)``) so it can
    be swapped into any block.  The sparse mask is built lazily per sequence
    length and cached.

    Parameters
    ----------
    d_model : int
        Model embedding dimension.
    num_heads : int
        Number of attention heads (must divide ``d_model``).
    dropout : float
        Attention dropout probability.
    pattern : str
        Sparse pattern (see :class:`SparseAttentionPattern`).
    window_size : int
        Band width for banded patterns.
    num_global_tokens : int
        Number of global tokens for global-aware patterns.
    block_size : int
        Block width for ``block_sparse``.
    causal : bool
        Whether the built mask is causal (default True).
    use_sdpa : bool
        Use ``torch.nn.functional.scaled_dot_product_attention`` (fused flash
        kernel on CUDA) instead of the explicit score matmul.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        pattern: str = SparseAttentionPattern.SLIDING_WINDOW.value,
        window_size: int = 128,
        num_global_tokens: int = 32,
        block_size: int = 64,
        causal: bool = True,
        use_sdpa: bool = False,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if pattern not in SPARSE_PATTERNS:
            raise ValueError(f"Unknown sparse pattern {pattern!r}; choose from {SPARSE_PATTERNS}.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.pattern = pattern
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        self.block_size = block_size
        self.causal = causal
        self.use_sdpa = use_sdpa
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Sparse mask cache keyed by (seq_len, device, dtype).
        self._mask_cache: dict[tuple, torch.Tensor] = {}

    def _mask_for(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, str(device))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        mask = sparse_attention_mask(
            self.pattern,
            seq_len,
            window_size=self.window_size,
            num_global_tokens=self.num_global_tokens,
            block_size=self.block_size,
            causal=self.causal,
            device=device,
        )
        self._mask_cache[key] = mask
        return mask

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        use_rope: bool = True,
    ) -> torch.Tensor:
        """
        x: (B, T, C)
        attn_mask: optional additive mask (1, 1, T, T) combined with the
            sparse pattern (e.g. an existing causal mask).
        Returns (B, T, C).
        """
        B, T, C = x.shape
        sparse = self._mask_for(T, x.device)

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if use_rope:
            q, k = self.rope(q, k)

        if self.use_sdpa:
            mask = sparse
            if attn_mask is not None:
                mask = mask + attn_mask
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + sparse
            if attn_mask is not None:
                scores = scores + attn_mask
            probs = F.softmax(scores, dim=-1)
            probs = self.dropout(probs)
            out = torch.matmul(probs, v)

        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))


def build_sparse_attention(
    pattern: str,
    d_model: int,
    num_heads: int,
    dropout: float = 0.1,
    *,
    window_size: int = 128,
    num_global_tokens: int = 32,
    block_size: int = 64,
    causal: bool = True,
    use_sdpa: bool = False,
) -> SparseSelfAttention:
    """Factory building :class:`SparseSelfAttention` for a requested pattern."""
    return SparseSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        pattern=pattern,
        window_size=window_size,
        num_global_tokens=num_global_tokens,
        block_size=block_size,
        causal=causal,
        use_sdpa=use_sdpa,
    )
