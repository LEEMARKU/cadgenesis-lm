"""tests/reasoning/test_knowledge_graph.py
=========================================
Unit tests for cadgenesis.reasoning.knowledge_graph.
"""

from __future__ import annotations

import pytest

from cadgenesis.reasoning.knowledge_graph import GraphEdge, GraphNode, KnowledgeGraph


@pytest.fixture
def graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g.add_node("aluminum", "Aluminum", node_type="material")
    g.add_node("machining", "Machining", node_type="process")
    g.add_node("6061", "Aluminum 6061", node_type="alloy")
    g.add_node("bracket", "Bracket", node_type="part")
    g.add_edge("machining", "aluminum", "processes")
    g.add_edge("aluminum", "6061", "has_alloy")
    g.add_edge("6061", "bracket", "used_in")
    return g


class TestNodes:
    def test_add_and_get(self, graph):
        assert graph.has_node("aluminum")
        node = graph.get_node("aluminum")
        assert node.label == "Aluminum"

    def test_duplicate_node(self):
        g = KnowledgeGraph()
        g.add_node("a")
        with pytest.raises(ValueError):
            g.add_node("a")

    def test_remove(self, graph):
        assert graph.remove_node("bracket")
        assert not graph.has_node("bracket")
        assert not graph.remove_node("bracket")

    def test_filter_by_type(self, graph):
        materials = graph.nodes(node_type="material")
        assert [n.id for n in materials] == ["aluminum"]

    def test_empty_label_defaults_to_id(self):
        node = GraphNode("x")
        assert node.label == "x"


class TestEdges:
    def test_add_edge_requires_nodes(self):
        g = KnowledgeGraph()
        g.add_node("a")
        with pytest.raises(KeyError):
            g.add_edge("a", "b", "rel")

    def test_has_edge(self, graph):
        assert graph.has_edge("machining", "aluminum", "processes")
        assert not graph.has_edge("aluminum", "machining", "processes")

    def test_edge_validation(self):
        with pytest.raises(ValueError):
            GraphEdge("a", "b", "")
        with pytest.raises(ValueError):
            GraphEdge("a", "b", "rel", weight=0)


class TestQuery:
    def test_neighbors(self, graph):
        neighbors = graph.neighbors("aluminum")
        assert sorted(target for target, _ in neighbors) == ["6061"]

    def test_neighbors_filter(self, graph):
        result = graph.neighbors("machining", relation="processes")
        assert [(target, e.relation) for target, e in result] == [("aluminum", "processes")]

    def test_predecessors(self, graph):
        assert sorted(graph.predecessors("bracket")) == ["6061"]

    def test_shortest_path(self, graph):
        path = graph.shortest_path("machining", "bracket")
        assert path == ["machining", "aluminum", "6061", "bracket"]

    def test_shortest_path_none(self, graph):
        g = KnowledgeGraph()
        g.add_node("a")
        g.add_node("b")
        assert g.shortest_path("a", "b") is None

    def test_find_related_depth_2(self, graph):
        related = graph.find_related("machining", max_depth=2)
        assert related == {"aluminum", "6061"}

    def test_query(self, graph):
        parts = list(graph.query(lambda n: n.node_type == "part"))
        assert [n.id for n in parts] == ["bracket"]


class TestPersistence:
    def test_json_round_trip(self, graph):
        text = graph.to_json()
        rebuilt = KnowledgeGraph.from_json(text)
        assert rebuilt.node_count == graph.node_count
        assert rebuilt.edge_count == graph.edge_count
        assert rebuilt.shortest_path("machining", "bracket") == graph.shortest_path(
            "machining", "bracket"
        )

    def test_to_dict_structure(self, graph):
        data = graph.to_dict()
        assert {"nodes", "edges"} <= set(data)


class TestStats:
    def test_counts(self, graph):
        assert graph.node_count == 4
        assert graph.edge_count == 3

    def test_relations(self, graph):
        assert graph.relations() == {"processes", "has_alloy", "used_in"}

    def test_degrees(self, graph):
        degrees = graph.degrees()
        assert degrees["aluminum"] == 2
