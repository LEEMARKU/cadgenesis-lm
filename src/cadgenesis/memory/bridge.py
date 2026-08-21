"""cadgenesis.memory.bridge
=========================
Semantic → neural memory bridge (v6.0, Pillar 6).

Connects the pure-Python semantic layer
(:class:`~cadgenesis.memory.memory_system.MemorySystem`, ``MemoryStore``) to the
torch neural bank (:class:`~cadgenesis.memory.memory_pools.LayerIntegratedMemorySystem`)
so that ``MemoryAttention`` can attend to *stored knowledge* instead of
random slot vectors.

:class:`SemanticMemoryBridge` renders retrieval hits into deterministic
d_model vectors (hash-bag embedding), lets callers write them into a neural
pool, and offers a ``combined_bank`` view that appends the rendered context to
the existing pool bank.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import torch

from cadgenesis.memory.retrieval import RetrievalResult

_WORD_RE = re.compile(r"[a-z0-9_]+")


class SemanticMemoryBridge:
    """Renders semantic hits into neural slot vectors for memory attention."""

    def __init__(self, d_model: int = 1024, seed: int = 0) -> None:
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        self.d_model = d_model
        self.seed = seed

    # ------------------------------------------------------------ embedding

    def embed_text(self, text: str) -> torch.Tensor:
        """Deterministic hash-bag embedding of ``text`` → (d_model,).

        Each token is hashed into a feature index; hits accumulate into a
        bag-of-words style vector that is L2-normalised, so it is stable
        across calls and requires no trained encoder.
        """
        vector = torch.zeros(self.d_model)
        tokens = _WORD_RE.findall(text.lower())
        for token in tokens:
            digest = hashlib.blake2b(f"{token}:{self.seed}".encode(), digest_size=8).digest()
            index = int.from_bytes(digest, "little") % self.d_model
            weight = (int.from_bytes(digest, "little") >> 16) / 65536.0 + 0.5
            vector[index] += weight
        norm = torch.linalg.vector_norm(vector)
        if norm > 1e-8:
            vector = vector / norm
        return vector

    def embed_entry(self, entry: Any) -> torch.Tensor:
        """Embed any record with a ``text()``/``content``/``metadata`` shape."""
        text = getattr(entry, "text", None)
        if text is not None:
            try:
                return self.embed_text(text())
            except TypeError:
                pass
        content = getattr(entry, "content", entry)
        metadata = getattr(entry, "metadata", None)
        parts = [str(content)]
        if metadata:
            parts.extend(str(v) for v in metadata.values())
        return self.embed_text(" ".join(parts))

    # -------------------------------------------------------------- render

    def to_vectors(
        self,
        result: RetrievalResult,
        top_k: int | None = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Render hits into (batch_size, k, d_model) slot vectors.

        Hits are importance/score ordered; each becomes one slot vector.
        When there are no hits, a single zero vector slot is returned so the
        attention path still has a tensor to attend to.
        """
        hits = result.hits if top_k is None else result.hits[:top_k]
        if not hits:
            return torch.zeros(batch_size, 1, self.d_model)
        vectors = torch.stack([self.embed_entry(hit.entry) for hit in hits])  # (k, d_model)
        return vectors.unsqueeze(0).expand(batch_size, -1, -1)

    # ------------------------------------------------------------- writing

    def write_pool(
        self,
        system: Any,
        result: RetrievalResult,
        pool_name: str = "working",
        max_slots: int | None = None,
    ) -> list[int]:
        """Write rendered hit vectors into a neural pool (no grad).

        Returns the slot indices that were overwritten.
        """
        pool = system.get_pool(pool_name)
        vectors = self.to_vectors(result, top_k=max_slots)
        k = vectors.shape[1]
        k = min(k, pool.num_slots)
        indices = torch.arange(k)
        pool.write_memory(indices, vectors[0, :k])
        return indices.tolist()

    def combined_bank(
        self,
        system: Any,
        result: RetrievalResult,
        batch_size: int = 1,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Append rendered context to the existing combined memory bank.

        Returns (batch_size, total_slots + k, d_model) so the model can
        attend to stored knowledge alongside the native pools.
        """
        bank = system.get_combined_memory_bank(batch_size)
        context = self.to_vectors(result, top_k=top_k, batch_size=batch_size)
        return torch.cat([bank, context], dim=1)

    def stats(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "seed": self.seed,
            "embedding": "hash-bag blake2b",
        }


__all__ = ["SemanticMemoryBridge"]
