"""cadgenesis.multimodal.encoders.text
====================================
Natural-language encoder (engineering prompts, conversational reasoning,
technical terminology, engineering specifications).

The encoder first builds a deterministic hashed bag-of-ngrams descriptor for
each text (dependency-free, so retrieval features are stable without any
downloaded model), then maps it through a small MLP into the raw feature
space consumed by the shared engineering embedding space.

It also accepts pre-tokenized id sequences (``encode_ids``) for pipelines
that tokenize with the Autonomous CAD Tokenizer first.
"""

from __future__ import annotations

import itertools
import re
from typing import Any, ClassVar

import torch
import torch.nn as nn

from cadgenesis.multimodal.common import Modality
from cadgenesis.multimodal.encoders.base import MultimodalEncoder

_WORD_RE = re.compile(r"[a-z0-9]+")


def _hash_token(token: str, bucket: int) -> int:
    """Deterministic string hash into ``bucket`` buckets (FNV-1a)."""
    h = 2166136261
    for byte in token.encode("utf-8"):
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF
    return h % bucket


def _ngram_tokens(text: str) -> list[str]:
    """Lowercased word tokens plus adjacent bigrams."""
    words = _WORD_RE.findall(text.lower())
    if not words:
        return []
    tokens: list[str] = list(words)
    tokens.extend(f"{a}_{b}" for a, b in itertools.pairwise(words))
    return tokens


def text_descriptor(text: str, vocab_size: int = 4096) -> torch.Tensor:
    """L1-normalised hashed bag-of-(word+bigram) descriptor for ``text``.

    Returns a ``(vocab_size,)`` tensor.  Two synonymous engineering sentences
    share many bigrams, so the descriptor is a meaningful, stable retrieval
    feature even before any learned parameters.
    """
    vec = torch.zeros(vocab_size, dtype=torch.float32)
    for token in _ngram_tokens(text):
        vec[_hash_token(token, vocab_size)] += 1.0
    norm = vec.sum()
    if norm > 0:
        vec /= norm
    return vec


class TextEncoder(MultimodalEncoder):
    """Encoder for the ``text`` modality."""

    modality: ClassVar[Modality] = Modality.TEXT

    def __init__(
        self,
        feature_dim: int = 512,
        vocab_size: int = 4096,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(feature_dim=feature_dim)
        if vocab_size < 1 or hidden_dim < 1:
            raise ValueError("vocab_size and hidden_dim must be positive")
        self.vocab_size = vocab_size
        self.net = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        self.id_embed = nn.Embedding(vocab_size, feature_dim, padding_idx=0)

    def descriptor(self, text: str) -> torch.Tensor:
        return text_descriptor(text, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, vocab_size) descriptor -> (B, feature_dim)."""
        if x.dim() != 2 or x.shape[-1] != self.vocab_size:
            raise ValueError(
                f"text encoder expects (B, {self.vocab_size}) descriptors; got {tuple(x.shape)}"
            )
        return self.net(x)

    def encode(self, inputs: Any) -> torch.Tensor:
        """Accepts a ``str``, a list of ``str``, or a ``(B, vocab_size)``
        descriptor tensor.  Returns ``(B, feature_dim)``."""
        if isinstance(inputs, torch.Tensor):
            return self.forward(inputs)
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        if not texts:
            raise ValueError("cannot encode an empty text batch")
        descriptors = torch.stack([self.descriptor(t) for t in texts])
        return self.forward(descriptors)

    def encode_ids(self, token_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Embed a ``(B, T)`` integer token-id sequence with a learned table.

        Returns a pooled ``(B, feature_dim)`` vector.  Useful when the prompt
        is already tokenized by the Autonomous CAD Tokenizer.
        """
        embedded = self.id_embed(token_ids)
        if mask is not None:
            embedded = embedded * mask.unsqueeze(-1)
            return embedded.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return embedded.mean(dim=1)


__all__ = ["TextEncoder", "text_descriptor"]
