"""tests/memory/test_memory_router.py
====================================
Unit tests for cross-pool memory routing.
"""

from __future__ import annotations

from cadgenesis.memory.cad_memory import CADMemory
from cadgenesis.memory.manufacturing_memory import ManufacturingMemory
from cadgenesis.memory.memory_router import MemoryRouter, RoutingDecision


def _build_router() -> MemoryRouter:
    cad = CADMemory(capacity=16)
    cad.remember_feature_tree("part:flange", [{"op": "extrude"}])
    mfg = ManufacturingMemory(capacity=16)
    mfg.remember_process("milling", {"max_tool_diameter": 20})
    return MemoryRouter(
        [cad, mfg],
        domain_keywords={
            "cad": {"feature", "brep", "extrude"},
            "manufacturing": {"machining", "tool", "milling"},
        },
    )


def test_route_ranks_pools():
    router = _build_router()
    decisions = router.route("milling tool limits")
    assert isinstance(decisions[0], RoutingDecision)
    assert decisions[0].pool == "manufacturing"


def test_best_pool():
    router = _build_router()
    assert router.best_pool("extrude feature") == "cad"


def test_best_pool_empty_router():
    router = MemoryRouter()
    assert router.best_pool("anything") is None


def test_retrieve_routed():
    router = _build_router()
    result = router.retrieve("milling")
    assert result.hits
    assert result.top.pool == "manufacturing"


def test_register_and_unregister():
    router = _build_router()
    assert "simulation" not in router.pool_names
    assert router.unregister("cad")
    assert "cad" not in router.pool_names


def test_summary():
    router = _build_router()
    summary = router.summary()
    assert "cad" in summary["affinity_keywords"]
