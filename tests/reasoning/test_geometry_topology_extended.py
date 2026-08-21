"""Tests for Pillar 7 geometry feature-dependency and topology adjacency reasoning."""

from __future__ import annotations

import pytest

from cadgenesis.reasoning import GeometryReasoner, Primitive, TopologyAnalyzer


def _feature_tree() -> list[dict]:
    return [
        {"id": "sketch1", "kind": "sketch"},
        {"id": "extrude1", "kind": "extrude", "depends_on": ["sketch1"]},
        {"id": "hole1", "kind": "hole", "depends_on": ["extrude1"]},
        {"id": "fillet1", "kind": "fillet", "depends_on": ["hole1"]},
    ]


def test_feature_dependencies_pairs() -> None:
    pairs = GeometryReasoner.feature_dependencies(_feature_tree())
    assert len(pairs) == 3
    assert pairs[0][0]["id"] == "sketch1"
    assert pairs[0][1]["id"] == "extrude1"


def test_validate_feature_dependencies_ok() -> None:
    validation = GeometryReasoner.validate_feature_dependencies(_feature_tree())
    assert validation.valid


def test_validate_dangling_dependency() -> None:
    features = [
        {"id": "a", "kind": "sketch"},
        {"id": "b", "kind": "extrude", "depends_on": ["missing"]},
    ]
    validation = GeometryReasoner.validate_feature_dependencies(features)
    assert not validation.valid
    assert any("missing" in m for m in validation.messages)


def test_validate_cycle_detection() -> None:
    features = [
        {"id": "a", "kind": "model", "depends_on": ["b"]},
        {"id": "b", "kind": "model", "depends_on": ["a"]},
    ]
    validation = GeometryReasoner.validate_feature_dependencies(features)
    assert not validation.valid
    assert any("cycle" in m for m in validation.messages)


def test_feature_order_topological() -> None:
    ok, order = GeometryReasoner.feature_order(_feature_tree())
    assert ok
    assert order == ["sketch1", "extrude1", "hole1", "fillet1"]


def test_feature_order_cycle_returns_false() -> None:
    features = [
        {"id": "a", "kind": "model", "depends_on": ["b"]},
        {"id": "b", "kind": "model", "depends_on": ["a"]},
    ]
    ok, order = GeometryReasoner.feature_order(features)
    assert not ok
    assert order == []


def test_geometric_consistency_ok() -> None:
    primitives = [
        Primitive("box", {"length": 2, "width": 2, "height": 2}, position=(0, 0, 0)),
        Primitive("box", {"length": 2, "width": 2, "height": 2}, position=(5, 5, 5)),
    ]
    validation = GeometryReasoner.geometric_consistency(primitives)
    assert validation.valid


def test_geometric_consistency_flags_interference() -> None:
    primitives = [
        Primitive("box", {"length": 10, "width": 10, "height": 10}, position=(0, 0, 0)),
        Primitive("box", {"length": 10, "width": 10, "height": 10}, position=(5, 5, 5)),
    ]
    validation = GeometryReasoner.geometric_consistency(primitives)
    assert not validation.valid
    assert any("overlaps" in m for m in validation.messages)


def test_geometric_consistency_tolerance() -> None:
    primitives = [
        Primitive("box", {"length": 10, "width": 10, "height": 10}, position=(0, 0, 0)),
        Primitive("box", {"length": 10, "width": 10, "height": 10}, position=(9, 9, 9)),
    ]
    validation = GeometryReasoner.geometric_consistency(primitives, allowed_interference=20.0)
    assert validation.valid


def test_tolerance_stack_worst_case() -> None:
    result = GeometryReasoner.tolerance_stack([(50.0, 0.1), (30.0, 0.2)])
    assert result["nominal"] == pytest.approx(80.0)
    assert result["worst"] == pytest.approx(0.3)
    assert result["rss"] == pytest.approx((0.1**2 + 0.2**2) ** 0.5)


def test_tolerance_stack_empty() -> None:
    result = GeometryReasoner.tolerance_stack([])
    assert result == {"nominal": 0.0, "worst": 0.0, "rss": 0.0}


# ---------------------------------------------------------------- topology


def _tetrahedron() -> list[tuple[int, ...]]:
    return [
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    ]


def test_adjacency_graph_tetrahedron() -> None:
    graph = TopologyAnalyzer.adjacency_graph(_tetrahedron())
    assert set(graph) == {0, 1, 2, 3}
    for neighbors in graph.values():
        assert len(neighbors) == 3


def test_adjacency_graph_disjoint() -> None:
    faces = [(0, 1, 2), (5, 6, 7)]
    graph = TopologyAnalyzer.adjacency_graph(faces)
    assert graph[0] == []
    assert graph[1] == []


def test_connectivity_reasoning_connected() -> None:
    result = TopologyAnalyzer.connectivity_reasoning(_tetrahedron())
    assert result["connected"] is True
    assert result["components"] == 1
    assert result["component_sizes"] == [4]


def test_connectivity_reasoning_disconnected() -> None:
    faces = [(0, 1, 2), (5, 6, 7)]
    result = TopologyAnalyzer.connectivity_reasoning(faces)
    assert result["connected"] is False
    assert result["components"] == 2


def test_connectivity_reasoning_empty_rejected() -> None:
    try:
        TopologyAnalyzer.connectivity_reasoning([])
        raised = False
    except ValueError:
        raised = True
    assert raised
