"""tests/memory/test_retrieval.py
================================
Unit tests for cross-pool retrieval.
"""

from __future__ import annotations

from cadgenesis.memory.cad_memory import CADMemory
from cadgenesis.memory.engineering_memory import EngineeringMemory
from cadgenesis.memory.retrieval import MemoryRetrieval, RetrievalResult


def _build_retriever() -> MemoryRetrieval:
    cad = CADMemory(capacity=16)
    cad.remember_feature_tree("part:flange", [{"op": "extrude", "name": "flange"}])
    eng = EngineeringMemory(capacity=16)
    eng.remember_standard("ISO-2768", "general tolerances for flanges")
    return MemoryRetrieval([cad, eng])


def test_retrieve_merges_pools():
    retriever = _build_retriever()
    result = retriever.retrieve("flange tolerances", top_k=4)
    assert isinstance(result, RetrievalResult)
    assert result.hits
    assert result.top is not None


def test_retrieve_pool_names_filter():
    retriever = _build_retriever()
    result = retriever.retrieve("flange", pool_names=["cad"])
    assert all(hit.pool == "cad" for hit in result.hits)


def test_retrieve_multi():
    retriever = _build_retriever()
    results = retriever.retrieve_multi(["flange", "tolerances"])
    assert set(results) == {"flange", "tolerances"}


def test_retrieve_unknown_pool_skipped():
    retriever = _build_retriever()
    result = retriever.retrieve("flange", pool_names=["nope", "cad"])
    assert result.hits


def test_register_and_unregister():
    retriever = _build_retriever()
    assert "simulation" not in retriever.pool_names
    retriever.register(CADMemory(capacity=8))
    assert retriever.unregister("cad")
    assert "cad" not in retriever.pool_names


def test_retrieve_empty_query():
    retriever = _build_retriever()
    result = retriever.retrieve("")
    assert result.hits == []


def test_summary():
    retriever = _build_retriever()
    summary = retriever.summary()
    assert set(summary["pools"]) == {"cad", "engineering"}
