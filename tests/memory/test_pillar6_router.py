"""tests/memory/test_pillar6_router.py
=====================================
Unit tests for the Pillar 6 contextual router extensions.
"""

from __future__ import annotations

from cadgenesis.memory.memory_common import MemoryStore
from cadgenesis.memory.memory_router import MemoryRouter


def build_router() -> MemoryRouter:
    stores = {
        "working": MemoryStore("working", capacity=16),
        "engineering": MemoryStore("engineering", capacity=16),
        "simulation": MemoryStore("simulation", capacity=16),
        "user": MemoryStore("user", capacity=16),
    }
    stores["engineering"].add("e1", "iso tolerance standard material")
    stores["simulation"].add("s1", "fea stress load safety factor")
    stores["user"].add("u1", "preference style profile")
    router = MemoryRouter(
        list(stores.values()),
        domain_keywords={
            "working": {"context", "current"},
            "engineering": {"tolerance", "standard", "iso"},
            "simulation": {"fea", "stress", "load"},
            "user": {"preference", "style", "user"},
        },
    )
    return router


def test_route_by_context_explicit_pool():
    router = build_router()
    ranked = router.route_by_context({"pool": "simulation"})
    assert ranked[0].pool == "simulation"
    assert ranked[0].score == 1.0


def test_route_by_context_boost():
    router = build_router()
    ranked = router.route_by_context({"text": "fea stress", "metadata": {"user": 5.0}})
    assert ranked[0].pool == "user"


def test_route_by_task_design():
    router = build_router()
    ranked = router.route_by_task("design")
    assert ranked[0].pool == "engineering"


def test_route_by_task_simulation():
    router = build_router()
    ranked = router.route_by_task("simulation")
    assert ranked[0].pool == "simulation"


def test_route_by_confidence_high_prefers_authoritative():
    router = build_router()
    ranked = router.route_by_confidence("tolerance standard", 0.95)
    assert ranked[0].pool == "engineering"


def test_route_by_confidence_low_prefers_working():
    router = build_router()
    ranked = router.route_by_confidence("tolerance standard", 0.05)
    assert ranked[0].pool == "working"


def test_route_by_agent_geometry():
    router = build_router()
    ranked = router.route_by_agent("geometry")
    assert ranked[0].pool == "engineering"


def test_route_by_agent_unknown_falls_back():
    router = build_router()
    ranked = router.route_by_agent("mystery-role")
    assert ranked
