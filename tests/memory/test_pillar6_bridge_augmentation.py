"""tests/memory/test_pillar6_bridge_augmentation.py
==================================================
Unit tests for the Pillar 6 semantic→neural bridge and augmentation helpers.
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.memory.augmentation import (
    ContextExpansion,
    MemoryAugmentedDecoding,
    MemoryRetrievalLayer,
    PersistentContextCache,
)
from cadgenesis.memory.bridge import SemanticMemoryBridge
from cadgenesis.memory.memory_common import MemoryStore
from cadgenesis.memory.memory_pools import LayerIntegratedMemorySystem


def _result():
    from cadgenesis.memory.retrieval import MemoryRetrieval

    store = MemoryStore("cad", capacity=16)
    store.add("p1", "tolerance for shaft machining", importance=0.9)
    store.add("p2", "stress safety factor", importance=0.8)
    return MemoryRetrieval([store]).retrieve("tolerance", top_k=2)


def test_embed_text_shape_and_determinism():
    bridge = SemanticMemoryBridge(d_model=64)
    v1 = bridge.embed_text("tolerance for shaft")
    v2 = bridge.embed_text("tolerance for shaft")
    assert v1.shape == (64,)
    assert torch.allclose(v1, v2)
    assert torch.linalg.vector_norm(v1) > 1e-8


def test_to_vectors_shape():
    bridge = SemanticMemoryBridge(d_model=32)
    vectors = bridge.to_vectors(_result(), top_k=2, batch_size=3)
    assert vectors.shape == (3, 2, 32)


def test_to_vectors_empty_result():
    bridge = SemanticMemoryBridge(d_model=16)
    from cadgenesis.memory.retrieval import RetrievalResult

    vectors = bridge.to_vectors(RetrievalResult(query="x"), batch_size=2)
    assert vectors.shape == (2, 1, 16)


def test_write_pool_writes_slots():
    bridge = SemanticMemoryBridge(d_model=16)
    system = LayerIntegratedMemorySystem(d_model=16)
    indices = bridge.write_pool(system, _result(), pool_name="working", max_slots=4)
    assert len(indices) == 2
    bank = system.get_pool("working").get_memory(1)
    assert not torch.allclose(bank[0, indices[0]], torch.zeros(16))


def test_combined_bank_appends_context():
    bridge = SemanticMemoryBridge(d_model=8)
    system = LayerIntegratedMemorySystem(d_model=8)
    bank = bridge.combined_bank(system, _result(), batch_size=1, top_k=2)
    assert bank.shape[1] == system.total_slots + 2


def test_retrieval_layer_forward_shape():
    layer = MemoryRetrievalLayer(d_model=32, num_heads=4)
    x = torch.randn(2, 5, 32)
    context = torch.randn(2, 3, 32)
    out = layer(x, context)
    assert out.shape == (2, 5, 32)


def test_retrieval_layer_rejects_bad_heads():
    with pytest.raises(ValueError):
        MemoryRetrievalLayer(d_model=32, num_heads=7)


def test_augmented_decoding_modes():
    layer = MemoryAugmentedDecoding(d_model=16)
    hidden = torch.randn(1, 4, 16)
    context = torch.randn(1, 2, 16)
    for mode in ("concat", "sum", "mean"):
        out = layer(hidden, context, mode=mode)
        assert out.shape == (1, 4, 16)
    with pytest.raises(ValueError):
        layer(hidden, context, mode="bogus")


def test_persistent_context_cache():
    cache = PersistentContextCache(max_entries=2)
    cache.push("a", content="ctx-a", vector=torch.randn(4))
    cache.push("b", content="ctx-b", vector=torch.randn(4))
    cache.push("c", content="ctx-c", vector=torch.randn(4))
    assert len(cache) == 2
    assert cache.get("a") is None  # evicted oldest
    assert cache.get("c") == "ctx-c"
    vectors = cache.vectors()
    assert vectors.shape == (2, 4)


def test_persistent_context_cache_clear():
    cache = PersistentContextCache()
    cache.push("a", content=1, vector=torch.randn(4))
    cache.clear()
    assert len(cache) == 0
    assert cache.vectors() is None


def test_context_expansion_trims_oldest():
    expansion = ContextExpansion(max_tokens=6)
    tokens = torch.randn(1, 4, 8)
    context = torch.randn(1, 3, 8)
    out = expansion.expand(tokens, context)
    assert out.shape == (1, 6, 8)
    assert not expansion.fitted(4, 3)
    assert expansion.fitted(2, 3)
