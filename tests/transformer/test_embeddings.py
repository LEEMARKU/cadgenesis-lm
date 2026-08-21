"""tests/transformer/test_embeddings.py
=====================================
Unit tests for cadgenesis.transformer.embeddings.
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.transformer.embeddings import (
    CombinedInputEmbedding,
    TokenEmbedding,
    TypeEmbedding,
)


class TestTokenEmbedding:
    def test_shape(self):
        emb = TokenEmbedding(vocab_size=100, d_model=32)
        x = torch.randint(0, 100, (2, 16))
        out = emb(x)
        assert out.shape == (2, 16, 32)

    def test_scale_is_sqrt_d_model(self):
        emb = TokenEmbedding(vocab_size=100, d_model=64)
        x = torch.ones(1, 1, dtype=torch.long)
        scale = torch.sqrt(torch.tensor(64.0))
        expected = scale * emb.embed.weight[1].reshape(1, 1, 64)
        assert torch.allclose(emb(x), expected)

    def test_validation(self):
        with pytest.raises(ValueError):
            TokenEmbedding(vocab_size=0, d_model=32)
        with pytest.raises(ValueError):
            TokenEmbedding(vocab_size=10, d_model=0)


class TestTypeEmbedding:
    def test_shape(self):
        emb = TypeEmbedding(num_types=5, d_model=32)
        x = torch.randint(0, 5, (2, 16))
        assert emb(x).shape == (2, 16, 32)

    def test_validation(self):
        with pytest.raises(ValueError):
            TypeEmbedding(num_types=0, d_model=32)


class TestCombinedInputEmbedding:
    def test_shape(self):
        emb = CombinedInputEmbedding(vocab_size=100, num_types=4, d_model=32)
        tok = torch.randint(0, 100, (2, 16))
        typ = torch.randint(0, 4, (2, 16))
        out = emb(tok, typ)
        assert out.shape == (2, 16, 32)

    def test_positional_disabled(self):
        emb = CombinedInputEmbedding(vocab_size=100, num_types=4, d_model=32, add_positional=False)
        tok = torch.ones(1, 4, dtype=torch.long)
        out = emb(tok)
        assert torch.allclose(out, emb.token_embedding(tok))

    def test_gradient_flow(self):
        emb = CombinedInputEmbedding(vocab_size=100, num_types=4, d_model=32)
        tok = torch.randint(0, 100, (2, 16))
        typ = torch.randint(0, 4, (2, 16))
        loss = emb(tok, typ).sum()
        loss.backward()
        assert emb.token_embedding.embed.weight.grad is not None
        assert emb.type_embedding.embed.weight.grad is not None
