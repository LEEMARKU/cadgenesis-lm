"""cadgenesis.transformer.modern_attention
=========================================
Modern attention mechanisms for CADGenesis-LM v2.0:

1. GroupedQueryAttention (GQA) — Ainslie et al. 2023: query heads share a
   reduced set of key/value heads to cut KV-cache memory while preserving
   most of the expressivity of full multi-head attention.
2. MultiHeadLatentAttention (MLA) — DeepSeek-V2/V3 style latent KV
   compression: keys and values are projected into a low-rank latent
   ``c_KV`` that is the sole thing cached at inference time, yielding
   dramatic KV-cache savings relative to standard MHA.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cadgenesis.transformer.positional import RotaryEmbedding

__all__ = ["GroupedQueryAttention", "MultiHeadLatentAttention"]


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).

    Only ``num_kv_heads`` distinct key/value heads are computed; each query
    head group of size ``num_heads // num_kv_heads`` attends to the same
    key/value head.  KV-cache memory drops by ``num_heads / num_kv_heads``
    versus standard multi-head attention while the query projections stay
    fully expressive (Ainslie et al., 2023).

    Parameters
    ----------
    d_model : int
        Model embedding dimension.  Must be divisible by ``num_heads``.
    num_heads : int
        Number of query heads.
    num_kv_heads : int | None
        Number of key/value heads.  Defaults to 1 (extreme GQA, one shared
        KV head for all queries); must be in ``[1, num_heads]`` and divide
        ``num_heads`` evenly.
    dropout : float
        Attention-dropout probability applied on the attention weights
        (active only while the module is in ``training`` mode).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        num_kv_heads = 1 if num_kv_heads is None else num_kv_heads
        if not 1 <= num_kv_heads <= num_heads:
            raise ValueError(
                f"num_kv_heads ({num_kv_heads}) must be in [1, num_heads] ({num_heads})."
            )
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        use_rope: bool = True,
    ) -> torch.Tensor:
        """
        x: (B, T, C)
        attn_mask: (1, 1, T, T) or (B, H, T, T), optional additive mask;
            when None the layer is *bidirectional* (no implicit causal mask —
            the encoder uses this; the decoder always passes a triangular
            mask explicitly).
        Returns: (B, T, C)
        """
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Memory-free head sharing: a zero-copy expand repeats each KV head
        # `num_heads // num_kv_heads` times along a fresh group axis, which is
        # then flattened back to the full head dimension.
        group_size = self.num_heads // self.num_kv_heads
        k = (
            k.unsqueeze(2)
            .expand(B, self.num_kv_heads, group_size, T, self.head_dim)
            .reshape(B, self.num_heads, T, self.head_dim)
        )
        v = (
            v.unsqueeze(2)
            .expand(B, self.num_kv_heads, group_size, T, self.head_dim)
            .reshape(B, self.num_heads, T, self.head_dim)
        )

        if use_rope:
            q, k = self.rope(q, k)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
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
        ``SelfAttention.forward_cached``.  Only the ``num_kv_heads`` distinct
        key/value heads are cached.
        """
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if use_rope:
            q, k = self.rope(q, k, position_offset=position_offset)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        T_k = k.shape[2]
        group_size = self.num_heads // self.num_kv_heads
        k_full = (
            k.unsqueeze(2)
            .expand(B, self.num_kv_heads, group_size, T_k, self.head_dim)
            .reshape(B, self.num_heads, T_k, self.head_dim)
        )
        v_full = (
            v.unsqueeze(2)
            .expand(B, self.num_kv_heads, group_size, T_k, self.head_dim)
            .reshape(B, self.num_heads, T_k, self.head_dim)
        )

        # NOTE: the KV cache enforces causality; ``is_causal=True`` with one
        # query would wrongly attend only the first key (tril(ones(1, S))).
        out = F.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C)), (
            k.detach(),
            v.detach(),
        )


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA), DeepSeek-V2/V3 style.

    Keys and values are jointly compressed into a shared low-rank latent
    ``c_KV = w_dk(x)`` of size ``kv_lora_rank``; both ``k_full = w_uk(c_KV)``
    and ``v = w_uv(c_KV)`` are recovered with up-projections.  Only ``c_KV``
    needs to be cached per token at inference time, so the KV-cache footprint
    is ``kv_lora_rank`` floats per token instead of ``2 * head_dim`` for
    standard MHA.

    Following DeepSeek, the query is split into a "nope" part (no positional
    information, concatenated with the un-rotated ``k_full``) and a "rope"
    part on which rotary embeddings are applied (using ``w_pe`` for the
    per-key rope component), so RoPE never contaminates the compressed
    latent key.

    Parameters
    ----------
    d_model : int
        Model embedding dimension.  Must be divisible by ``num_heads``.
    num_heads : int
        Number of attention heads.
    kv_lora_rank : int
        Dimension of the compressed KV latent ``c_KV``.
    qk_rope_head_dim : int
        Per-head dimension of the RoPE-applied query/key component.
        Must be ``<= head_dim``; the remainder ``head_dim - qk_rope_head_dim``
        is the un-rotated "nope" component.
    dropout : float
        Attention-dropout probability applied on the attention weights
        (active only while the module is in ``training`` mode).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        kv_lora_rank: int = 64,
        qk_rope_head_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        head_dim = d_model // num_heads
        if not 0 < qk_rope_head_dim <= head_dim:
            raise ValueError(
                f"qk_rope_head_dim ({qk_rope_head_dim}) must be in (0, head_dim] ({head_dim})."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = head_dim - qk_rope_head_dim
        self.dropout_p = dropout

        self.w_dk = nn.Linear(d_model, kv_lora_rank)
        self.w_uk = nn.Linear(kv_lora_rank, num_heads * head_dim, bias=False)
        self.w_uv = nn.Linear(kv_lora_rank, num_heads * head_dim, bias=False)
        self.w_q_nope = nn.Linear(d_model, num_heads * self.qk_nope_head_dim)
        self.w_q_rope = nn.Linear(d_model, num_heads * qk_rope_head_dim)
        self.w_pe = nn.Linear(d_model, num_heads * qk_rope_head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(qk_rope_head_dim)

        self.last_kv_latent: torch.Tensor | None = None

    def kv_cache_savings_ratio(self) -> float:
        """
        Fraction of KV-cache bytes saved versus standard MHA (v and k each
        cache ``head_dim`` floats per head per token; MLA caches only the
        ``kv_lora_rank`` latent).  Returns ``1 - kv_lora_rank / (2 * head_dim)``.
        """
        return 1.0 - self.kv_lora_rank / (2.0 * self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        use_rope: bool = True,
    ) -> torch.Tensor:
        """
        x: (B, T, C)
        attn_mask: (1, 1, T, T) or (B, H, T, T), optional additive mask;
            when None the layer is *bidirectional* (no implicit causal mask).
        Returns: (B, T, C)
        """
        B, T, C = x.shape

        c = self.w_dk(x)  # (B, T, kv_lora_rank) — the cached latent.
        self.last_kv_latent = c.detach()

        k_full = self.w_uk(c).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # w_uk projects into the full un-rotated key space; only the first
        # ``head_dim - qk_rope_head_dim`` dimensions pair with ``q_nope`` —
        # the tail is superseded by the rope component from ``w_pe``.
        k_nope = k_full[..., : self.qk_nope_head_dim]
        v = self.w_uv(c).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_rope = self.w_pe(x).view(B, T, self.num_heads, self.qk_rope_head_dim).transpose(1, 2)
        q_nope = self.w_q_nope(x).view(B, T, self.num_heads, self.qk_nope_head_dim).transpose(1, 2)
        q_rope = self.w_q_rope(x).view(B, T, self.num_heads, self.qk_rope_head_dim).transpose(1, 2)

        if use_rope:
            q_rope, k_rope = self.rope(q_rope, k_rope)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))

    def forward_cached(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        use_rope: bool = True,
        past_kv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Incremental forward for one new token (B, 1, C).

        ``past_kv`` is ``(k_nope, k_rope, v)`` (the already-up-projected key
        and value tensors).  Only ``k_nope``/``k_rope``/``v`` are cached — the
        per-token KV latent ``c_KV`` (see :meth:`kv_cache_savings_ratio`) is
        what MLA uses to *represent* those tensors compactly.
        """
        B, T, C = x.shape

        c = self.w_dk(x)  # (B, 1, kv_lora_rank)
        self.last_kv_latent = c.detach()

        k_full = self.w_uk(c).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_nope = k_full[..., : self.qk_nope_head_dim]
        v = self.w_uv(c).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_rope = self.w_pe(x).view(B, T, self.num_heads, self.qk_rope_head_dim).transpose(1, 2)
        q_nope = self.w_q_nope(x).view(B, T, self.num_heads, self.qk_nope_head_dim).transpose(1, 2)
        q_rope = self.w_q_rope(x).view(B, T, self.num_heads, self.qk_rope_head_dim).transpose(1, 2)

        if use_rope:
            q_rope, k_rope = self.rope(q_rope, k_rope, position_offset=position_offset)

        if past_kv is not None:
            k_nope = torch.cat([past_kv[0], k_nope], dim=2)
            k_rope = torch.cat([past_kv[1], k_rope], dim=2)
            v = torch.cat([past_kv[2], v], dim=2)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # NOTE: the KV cache enforces causality; ``is_causal=True`` with one
        # query would wrongly attend only the first key (tril(ones(1, S))).
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C)), (
            k_nope.detach(),
            k_rope.detach(),
            v.detach(),
        )
