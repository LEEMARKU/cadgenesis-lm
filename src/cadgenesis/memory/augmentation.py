"""cadgenesis.memory.augmentation
================================
Transformer memory augmentation (v6.0, Pillar 6).

Composable torch helpers that layer *stored knowledge* onto the existing
transformer without changing its forward contract:

* :class:`MemoryRetrievalLayer` — an ``nn.Module`` that attends over semantic
  retrieval vectors, producing a fixed-size context the model can consume.
* :class:`MemoryAugmentedDecoding` — prepends/concatenates retrieved context
  onto input hidden states at decode time.
* :class:`PersistentContextCache` — keeps a running key/value context across
  calls so multi-turn sessions share state.
* :class:`ContextExpansion` — grows a fixed window by appending retrieved
  context and dropping the oldest prefix.
"""

from __future__ import annotations

import uuid
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from cadgenesis.memory.bridge import SemanticMemoryBridge
from cadgenesis.memory.retrieval import RetrievalResult


class MemoryRetrievalLayer(nn.Module):
    """Attends over retrieval vectors → single context vector per position."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by num_heads {num_heads}")
        self.query_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = d_model**-0.5

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (B, T, C); context: (B, M, C) → (B, T, C)."""
        B, T, _ = x.shape
        head_dim = self.d_model // self.num_heads
        q = self.query_proj(x).view(B, T, self.num_heads, head_dim).transpose(1, 2)  # (B, H, T, dh)
        v = (
            self.value_proj(context).view(B, -1, self.num_heads, head_dim).transpose(1, 2)
        )  # (B, H, M, dh)
        scores = torch.matmul(q, v.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, H, T, dh)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)

    def retrieve_and_attend(
        self,
        x: torch.Tensor,
        bridge: SemanticMemoryBridge,
        result: RetrievalResult,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Convenience: render retrieval hits and attend to them."""
        context = bridge.to_vectors(result, top_k=top_k, batch_size=x.shape[0])
        return self.forward(x, context)


class MemoryAugmentedDecoding(nn.Module):
    """Concatenates retrieved context onto decode inputs (augmentation)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.compress = nn.Linear(2 * d_model, d_model)

    def forward(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        mode: str = "concat",
    ) -> torch.Tensor:
        """Augment ``hidden`` (B, T, C) with ``context`` (B, M, C).

        ``mode`` ∈ {concat, sum, mean}.
        """
        if mode not in {"concat", "sum", "mean"}:
            raise ValueError(f"unknown mode {mode!r}")
        if mode == "concat":
            pooled = context.mean(dim=1, keepdim=True)  # (B, 1, C)
            repeated = pooled.expand(-1, hidden.shape[1], -1)
            return self.compress(torch.cat([hidden, repeated], dim=-1))
        if mode == "sum":
            return hidden + context.mean(dim=1, keepdim=True)
        return hidden * 0.5 + context.mean(dim=1, keepdim=True) * 0.5


class PersistentContextCache:
    """Maintains running context across calls (session KV cache)."""

    def __init__(self, max_entries: int = 64, d_model: int | None = None) -> None:
        self.max_entries = max_entries
        self.d_model = d_model
        self._entries: list[dict[str, Any]] = []

    def push(
        self,
        key: str | None = None,
        content: Any = None,
        vector: torch.Tensor | None = None,
    ) -> str:
        """Add a context fragment; returns its id."""
        if self.d_model is None and vector is not None:
            self.d_model = vector.shape[-1]
        entry_id = key or uuid.uuid4().hex[:12]
        self._entries.append({"id": entry_id, "content": content, "vector": vector})
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)
        return entry_id

    def get(self, entry_id: str) -> Any | None:
        for entry in self._entries:
            if entry["id"] == entry_id:
                return entry["content"]
        return None

    def vectors(self) -> torch.Tensor | None:
        """Stack all cached vectors → (M, C); None when empty."""
        vectors = [entry["vector"] for entry in self._entries if entry["vector"] is not None]
        if not vectors:
            return None
        return torch.stack(vectors)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()


class ContextExpansion:
    """Expands a fixed window by appending context and dropping oldest tokens."""

    def __init__(self, max_tokens: int) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}")
        self.max_tokens = max_tokens

    def expand(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Append ``context`` to ``tokens`` and trim to ``max_tokens``.

        tokens: (B, T, C); context: (B, M, C) → (B, max_tokens, C).
        The head of the token sequence is dropped first.
        """
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        merged = torch.cat([tokens, context], dim=1)
        if merged.shape[1] <= self.max_tokens:
            return merged
        return merged[:, -self.max_tokens :]

    def fitted(self, used_tokens: int, context_tokens: int) -> bool:
        return used_tokens + context_tokens <= self.max_tokens


__all__ = [
    "ContextExpansion",
    "MemoryAugmentedDecoding",
    "MemoryRetrievalLayer",
    "PersistentContextCache",
]
