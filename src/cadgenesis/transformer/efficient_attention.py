"""
cadgenesis.transformer.efficient_attention
====================================
Efficient Attention Optimizations for CADGenesis-LM v2.0.

Provides drop-in replacements for :class:`cadgenesis.transformer.attention.SelfAttention`
with identical I/O contracts (``forward(x, attn_mask=None, use_rope=True)`` →
``(B, T, d_model)``) so any of them can be swapped into the attention mixture
without changing the surrounding architecture:

1. ``SDPASelfAttention`` — wraps :func:`torch.nn.functional.scaled_dot_product_attention`.
   On CUDA this transparently uses the fused **FlashAttention** /
   **memory-efficient** backends (``torch.backends.cuda.*_sdp_enabled``) and on
   CPU falls back to the math kernel.  The backend actually used for the last
   forward pass is recorded on ``self.last_backend``.
2. ``LinearAttention`` — Performer-style random-feature (FAVOR+ / RFA) linear
   attention with O(T · N) time and memory where ``N`` is the number of random
   features.  Fully causal via cumulative sums.

Factory
-------
``build_self_attention(backend, d_model, num_heads, dropout)`` maps the
configuration strings ``"math" | "sdpa" | "flash" | "linear"`` to the
appropriate module.  ``"flash"`` forces the SDPA module (the fused backend is
only engaged on GPU hardware that supports it; it degrades gracefully).

Complexity
----------
    SDPASelfAttention :  O(T² · C)  (same FLOPs as math, far lower I/O on GPU)
    LinearAttention   :  O(T · N · C)   (linear in sequence length)
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cadgenesis.transformer.positional import RotaryEmbedding

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Public backend identifiers
# --------------------------------------------------------------------------- #

MATH_BACKEND = "math"
SDPA_BACKEND = "sdpa"
FLASH_BACKEND = "flash"
LINEAR_BACKEND = "linear"
GQA_BACKEND = "gqa"
MLA_BACKEND = "mla"

BACKENDS = (MATH_BACKEND, SDPA_BACKEND, FLASH_BACKEND, LINEAR_BACKEND, GQA_BACKEND, MLA_BACKEND)


def _infer_sdpa_backend() -> str:
    """Report the fused SDPA kernel that would be used on this device."""
    if not torch.cuda.is_available():
        return "math"
    if getattr(torch.backends.cuda, "flash_sdp_enabled", lambda: False)():
        return "flash"
    if getattr(torch.backends.cuda, "mem_efficient_sdp_enabled", lambda: False)():
        return "mem_efficient"
    return "math"


# --------------------------------------------------------------------------- #
# Efficient self-attention modules
# --------------------------------------------------------------------------- #


class SDPASelfAttention(nn.Module):
    """
    Multi-head self-attention using ``torch.nn.functional.scaled_dot_product_attention``.

    Identical input/output contract to :class:`SelfAttention`; supports RoPE,
    arbitrary attention masks and native causal masking.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout
        self.rope = RotaryEmbedding(self.head_dim)

        # Last kernel actually used (diagnostics / logging).
        self.last_backend: str = "n/a"

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        use_rope: bool = True,
    ) -> torch.Tensor:
        """
        x: (B, T, C)
        attn_mask: (1, 1, T, T) or (B, H, T, T), optional additive mask.
            When None the layer is *bidirectional* (no implicit causal mask);
            callers wanting causality must pass a triangular additive mask.
        """
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if use_rope:
            q, k = self.rope(q, k)

        is_causal = False
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
        self.last_backend = _infer_sdpa_backend()
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))

    def forward_cached(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        use_rope: bool = True,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Incremental forward for one new token (B, 1, C); see
        ``SelfAttention.forward_cached``.  With the KV cache the query sits at
        the last position, so causal masking (``is_causal=True``) attends every
        cached key exactly.
        """
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if use_rope:
            q, k = self.rope(q, k, position_offset=position_offset)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        # NOTE: the KV cache already enforces causality (every cached key
        # precedes the current query), so full attention is correct here.
        # PyTorch's ``is_causal=True`` with a single query would instead build
        # ``tril(ones(1, S))`` and attend only the first key.
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        self.last_backend = _infer_sdpa_backend()
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C)), (
            k.detach(),
            v.detach(),
        )


class LinearAttention(nn.Module):
    """
    Performer-style random-feature (RFA) linear attention.

    Attention is approximated with positive random features::

        Attn(x) ≈ φ(Q) · ( Σ_j φ(K)_j ⊗ V_j ) / ( φ(Q) · Σ_j φ(K)_j )

    where ``φ`` is the FAVOR+ kernel feature map (default 64 random features).
    A causal mask is implemented exactly via cumulative sums along the time
    axis, so runtime is O(T · N · C) instead of O(T² · C).

    Note: linear attention is a *regulariser* approximation — for small
    sequences the quadratic ``math`` backend is more accurate.  It shines at
    long sequences where the quadratic memory would be prohibitive.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        num_random_features: int = 64,
        random_seed: int = 0,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if num_random_features < 1:
            raise ValueError("num_random_features must be >= 1.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_random_features = num_random_features
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Orthogonalised random projection: (num_random_features, head_dim).
        gen = torch.Generator().manual_seed(random_seed)
        r = torch.randn(num_random_features, self.head_dim, generator=gen)
        q_r, _ = torch.linalg.qr(r.T)  # (head_dim, num_random_features)
        self._proj: torch.Tensor
        self.register_buffer("_proj", q_r.T.unsqueeze(0).unsqueeze(0), persistent=False)

    # ------------------------------------------------------------- feature map

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """
        FAVOR+ style positive random features.
        x: (B, H, T, head_dim) → (B, H, T, num_random_features)
        """
        _B, _H, _T, _D = x.shape
        proj = self._proj.to(x.device)  # (1, 1, N, D)
        # exp(x · r / sqrt(N)) · exp(-||x||²/2)  → positive, unbiased-ish kernel.
        x_norm = (x @ proj.transpose(-2, -1)) / math.sqrt(self.num_random_features)
        decay = -0.5 * (x * x).sum(dim=-1, keepdim=True)
        return torch.exp(x_norm + decay) / math.sqrt(self.num_random_features)

    # --------------------------------------------------------------- forward

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        use_rope: bool = True,
    ) -> torch.Tensor:
        """
        x: (B, T, C)
        attn_mask: (1, 1, T, T) additive mask.  None → bidirectional (encoder);
            a causal triangular mask → exact causal via cumulative sums;
            a restrictive non-causal mask (sequence packing) → exact quadratic
            fallback; an all-zeros mask → bidirectional.
        """
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if use_rope:
            q, k = self.rope(q, k)

        # Mask semantics (consistent with the other backends):
        #  * None (encoder)  → *bidirectional*: attend to all keys (plain sum
        #    over the time axis, no cumsum).
        #  * causal triangular mask (decoder) → exact causality via cumsum.
        #  * restrictive non-causal mask (sequence packing) → exact quadratic
        #    fallback (cannot be expressed with prefix sums).
        #  * all-zeros mask → restricts nothing, treated as bidirectional.
        if attn_mask is not None and not self._is_causal(attn_mask, T):
            if torch.isneginf(attn_mask).any():
                return self._masked_forward(q, k, v, attn_mask)
            attn_mask = None
        causal = attn_mask is not None

        phi_q = self._phi(q)  # (B, H, T, N)
        phi_k = self._phi(k)  # (B, H, T, N)

        # kv-sums over time → O(T · N · C): cumulative (causal) or total.
        kv = torch.einsum("bhTn,bhTm->bhTnm", phi_k, v)  # (B, H, T, N, D)
        if causal:
            kv = kv.cumsum(dim=2)
            denom = phi_k.sum(dim=-1).cumsum(dim=2).clamp_min(1e-9)  # (B, H, T)
        else:
            kv = kv.sum(dim=2, keepdim=True)
            denom = phi_k.sum(dim=-1).sum(dim=2, keepdim=True).clamp_min(1e-9)  # (B, H, 1)

        num = torch.einsum("bhTn,bhTnd->bhTd", phi_q, kv)  # (B, H, T, D)
        out = num / denom.unsqueeze(-1)
        out = self.dropout(out)

        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))

    def _masked_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Exact quadratic attention with an additive mask (masked fallback).

        q/k/v: (B, H, T, head_dim) after RoPE; attn_mask: additive
        (1, 1, T, T) / (B, H, T, T) / (B, 1, T, T) mask.  Used only when a
        restrictive non-causal mask is supplied (sequence packing).
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + attn_mask
        probs = torch.softmax(scores, dim=-1)
        fully_masked = torch.isneginf(scores).all(dim=-1, keepdim=True)
        if fully_masked.any():
            probs = probs.masked_fill(fully_masked, 0.0)
        probs = self.dropout(probs)
        out = torch.matmul(probs, v)
        return self.out_proj(out.transpose(1, 2).contiguous().view(scores.shape[0], scores.shape[2], self.d_model))

    def forward_cached(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        use_rope: bool = True,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Incremental forward for one new token (B, 1, C) using linear-attention
        prefix accumulators.  ``past_kv`` is ``(kv_cum, denom)``; since the
        cumulative sums are additive, appending a single new token is exact.
        """
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if use_rope:
            q, k = self.rope(q, k, position_offset=position_offset)

        phi_q = self._phi(q)
        phi_k = self._phi(k)

        kv_t = torch.einsum("bhTn,bhTm->bhTnm", phi_k, v)  # (B, H, 1, N, D)
        denom_t = phi_k.sum(dim=-1)  # (B, H, 1)

        if past_kv is not None:
            kv_cum = past_kv[0] + kv_t
            denom = past_kv[1] + denom_t
        else:
            kv_cum, denom = kv_t, denom_t

        num = torch.einsum("bhTn,bhTnd->bhTd", phi_q, kv_cum)
        out = (num / denom.unsqueeze(-1)).clamp_min(-1e4)
        out = self.dropout(out)

        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C)), (
            kv_cum.detach(),
            denom.detach(),
        )

    @staticmethod
    def _is_causal(attn_mask: torch.Tensor | None, T: int) -> bool:
        """Detect an exact causal triangular additive mask.

        A causal mask has ``-inf`` on every strictly-upper-triangular entry
        and finite (zero) values on and below the diagonal.
        """
        if attn_mask is None:
            return False
        m = attn_mask
        if m.ndim < 2:
            return False
        # Reduce over leading (batch/head) dims; keep only the trailing (T, T).
        neg_inf = torch.isneginf(m)
        if m.ndim > 2:
            neg_inf = neg_inf.all(dim=tuple(range(m.ndim - 2)))
        if neg_inf.ndim != 2 or neg_inf.shape[-2:] != (T, T):
            return False
        triu_idx = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=m.device), diagonal=1
        )
        # Every strictly-upper-triangular entry must be -inf and the rest finite.
        return bool(neg_inf[triu_idx].all() and (~neg_inf[~triu_idx]).all())


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def build_self_attention(
    backend: str,
    d_model: int,
    num_heads: int,
    dropout: float = 0.1,
    num_random_features: int = 64,
    num_kv_heads: int | None = None,
    kv_lora_rank: int = 64,
    qk_rope_head_dim: int = 64,
) -> nn.Module:
    """
    Build a self-attention module for the requested backend.

    backend: one of ``"math" | "sdpa" | "flash" | "linear" | "gqa" | "mla"``
        * ``math``   → legacy quadratic ``SelfAttention`` (default; identical to
          the pre-upgrade behaviour).
        * ``sdpa``   → :class:`SDPASelfAttention` (fused kernel when available).
        * ``flash``  → :class:`SDPASelfAttention` with the SDPA backend (the
          fused flash kernel is engaged automatically on supported GPUs).
        * ``linear`` → :class:`LinearAttention` (Performer-style, O(T)).
        * ``gqa``    → :class:`GroupedQueryAttention` (Ainslie et al. 2023);
          ``num_kv_heads`` defaults to 1 (extreme GQA).
        * ``mla``    → :class:`MultiHeadLatentAttention` (DeepSeek-V2/V3);
          ``kv_lora_rank`` / ``qk_rope_head_dim`` control KV compression.
    """
    from cadgenesis.transformer.attention import SelfAttention  # deferred, avoid cycle

    if backend not in BACKENDS:
        raise ValueError(f"Unknown attention backend {backend!r}; choose from {BACKENDS}.")
    if backend == MATH_BACKEND:
        return SelfAttention(d_model, num_heads, dropout)
    if backend in (SDPA_BACKEND, FLASH_BACKEND):
        return SDPASelfAttention(d_model, num_heads, dropout)
    if backend == LINEAR_BACKEND:
        return LinearAttention(d_model, num_heads, dropout, num_random_features)
    if backend == GQA_BACKEND:
        from cadgenesis.transformer.modern_attention import GroupedQueryAttention

        return GroupedQueryAttention(d_model, num_heads, num_kv_heads, dropout)
    if backend == MLA_BACKEND:
        from cadgenesis.transformer.modern_attention import MultiHeadLatentAttention

        return MultiHeadLatentAttention(
            d_model,
            num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            dropout=dropout,
        )
    raise AssertionError(f"Unreachable backend {backend!r}")  # pragma: no cover
