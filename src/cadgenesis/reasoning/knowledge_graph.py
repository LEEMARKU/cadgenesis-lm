"""cadgenesis.reasoning.knowledge_graph
======================================
Knowledge graph for engineering knowledge (ISO/ASME/DIN standards, materials,
processes, token relations).

A simple directed, weighted graph: nodes are engineering concepts (materials,
standards, features, processes), edges are typed relations (``requires``,
``used_in``, ``constrains``, ``alternative_to``, …).  Supports neighbourhood
queries, shortest paths, related-concept expansion and JSON persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphNode:
    """A concept node in the knowledge graph."""

    id: str
    label: str = ""
    node_type: str = "concept"
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("node id must be non-empty")
        if not self.label:
            self.label = self.id


@dataclass
class GraphEdge:
    """A typed, weighted directed edge between two nodes."""

    source: str
    target: str
    relation: str
    weight: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ValueError("edge endpoints must be non-empty")
        if not self.relation:
            raise ValueError("edge relation must be non-empty")
        if self.weight <= 0:
            raise ValueError("edge weight must be positive")


class KnowledgeGraph:
    """Adjacency-backed directed knowledge graph with JSON persistence.

    Supports requirement traceability (``add_requirement``,
    ``find_requirements_by_feature``, ``find_requirements_by_op``,
    ``requirement_traceability_path``) in addition to the standard graph
    operations (``add_node``, ``add_edge``, ``remove_node``, query/methods).
    """

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._out: dict[str, list[GraphEdge]] = {}
        self._in: dict[str, list[GraphEdge]] = {}

    # ------------------------------------------------------------------ nodes

    def add_node(
        self,
        node_id: str,
        label: str = "",
        node_type: str = "concept",
        attributes: dict[str, Any] | None = None,
    ) -> GraphNode:
        node = GraphNode(node_id, label, node_type, attributes or {})
        if node_id in self._nodes:
            raise ValueError(f"node {node_id!r} already exists")
        self._nodes[node_id] = node
        self._out.setdefault(node_id, [])
        self._in.setdefault(node_id, [])
        return node

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def nodes(self, node_type: str | None = None) -> list[GraphNode]:
        """Return list of nodes, optionally filtered by *node_type*."""
        if node_type is None:
            return list(self._nodes.values())
        return [
            n for n in self._nodes.values() if n.node_type == node_type
        ]

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self._nodes:
            return False
        for edge in list(self._out.get(node_id, [])):
            self._in[edge.target].remove(edge)
        for edge in list(self._in.get(node_id, [])):
            self._out[edge.source].remove(edge)
        del self._nodes[node_id]
        del self._out[node_id]
        del self._in[node_id]
        return True

    # -------------------------------------------------------------- edges

    def has_edge(
        self, source: str, target: str, relation: str
    ) -> bool:
        """Return True if a directed edge *source* → *target* with *relation* exists."""
        if source not in self._nodes or target not in self._nodes:
            return False
        for edge in self._out.get(source, []):
            if edge.target == target and edge.relation == relation:
                return True
        return False

    def add_edge(
        self, source: str, target: str, relation: str, weight: float = 1.0
    ) -> GraphEdge:
        """Add a typed directed edge from *source* to *target*.

        Parameters
        ----------
        source : str
            Node ID of the edge source.
        target : str
            Node ID of the edge target.
        relation : str
            Edge relation name (e.g. ``"requires"``, ``"used_in"``).
        weight : float
            Edge weight (default ``1.0``).

        Returns
        -------
        GraphEdge
            The created edge.
        """
        if source not in self._nodes:
            raise KeyError(f"source node {source!r} does not exist")
        if target not in self._nodes:
            raise KeyError(f"target node {target!r} does not exist")
        edge = GraphEdge(source=source, target=target, relation=relation, weight=weight)
        self._out.setdefault(source, []).append(edge)
        self._in.setdefault(target, []).append(edge)
        return edge

    # -------------------------------------------------------------- neighbors

    def neighbors(
        self, node_id: str, relation: str | None = None
    ) -> list[tuple[str, GraphEdge]]:
        """Return list of (target_id, edge) for outgoing edges.

        Parameters
        ----------
        node_id : str
            Node ID to query.
        relation : str
            Optional relation filter; only edges with this relation are returned.
        """
        if node_id not in self._nodes:
            return []
        result: list[tuple[str, GraphEdge]] = []
        for edge in self._out.get(node_id, []):
            if relation is None or edge.relation == relation:
                result.append((edge.target, edge))
        return result

    def predecessors(self, node_id: str) -> list[str]:
        """Return list of predecessor node IDs (incoming edges)."""
        if node_id not in self._nodes:
            return []
        return [edge.source for edge in self._in.get(node_id, [])]

    # -------------------------------------------------------------- paths

    def shortest_path(self, source: str, target: str) -> list[str] | None:
        """Return shortest path from *source* to *target* using BFS, or None."""
        if source not in self._nodes or target not in self._nodes:
            return None
        if source == target:
            return [source]

        visited: set[str] = {source}
        queue: list[list[str]] = [[source]]
        while queue:
            path = queue.pop(0)
            last = path[-1]
            if last == target:
                return path
            for edge in self._out.get(last, []):
                nxt = edge.target
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(path + [nxt])
        return None

    def find_related(self, node_id: str, max_depth: int) -> set[str]:
        """Return set of node IDs reachable from *node_id* within *max_depth* hops.

        The starting node is excluded from the result unless max_depth >= 1
        and it is revisited via a cycle.
        """
        if node_id not in self._nodes or max_depth < 0:
            return set()
        if max_depth == 0:
            return set()

        visited: set[str] = set()
        frontier: list[str] = [node_id]
        depth = 0
        while frontier and depth < max_depth:
            next_frontier: list[str] = []
            for curr in frontier:
                for edge in self._out.get(curr, []):
                    nxt = edge.target
                    if nxt not in visited:
                        visited.add(nxt)
                        next_frontier.append(nxt)
            frontier = next_frontier
            depth += 1
        return visited

    # -------------------------------------------------------------- query

    def query(self, predicate) -> list[GraphNode]:
        """Return list of nodes matching *predicate* function."""
        return [node for node in self._nodes.values() if predicate(node)]

    # -------------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        """Return serializable representation of the graph."""
        return {
            "nodes": {
                nid: {"label": n.label, "node_type": n.node_type, "attributes": n.attributes}
                for nid, n in self._nodes.items()
            },
            "edges": {
                i: {"source": e.source, "target": e.target, "relation": e.relation, "weight": e.weight}
                for i, edges in self._out.items()
                for e in edges
            },
        }

    def to_json(self) -> str:
        """Serialize the graph to JSON."""
        import json
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> KnowledgeGraph:
        """Deserialize a graph from JSON text."""
        import json
        data = json.loads(text)
        g = cls()
        for nid, ndata in data.get("nodes", {}).items():
            g.add_node(
                nid,
                label=ndata.get("label", nid),
                node_type=ndata.get("node_type", "concept"),
                attributes=ndata.get("attributes", {}),
            )
        for i, edata in data.get("edges", {}).items():
            g.add_edge(
                edata["source"],
                edata["target"],
                edata["relation"],
                weight=edata.get("weight", 1.0),
            )
        return g

    # -------------------------------------------------------------- stats

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self._out.values())

    def relations(self) -> set[str]:
        """Return set of relation names used in the graph."""
        rels: set[str] = set()
        for edges in self._out.values():
            for edge in edges:
                rels.add(edge.relation)
        return rels

    def degrees(self) -> dict[str, int]:
        """Return dict of node ID -> total degree (in + out)."""
        deg: dict[str, int] = {}
        for nid in self._nodes:
            deg[nid] = len(self._out.get(nid, [])) + len(self._in.get(nid, []))
        return deg

    # -------------------------------------------------------------- requirements

    def add_requirement(
        self,
        req_id: str,
        label: str,
        description: str = "",
        related_features: list[str] | None = None,
        related_ops: list[str] | None = None,
    ) -> GraphNode:
        """Add a requirement node to the graph.

        Parameters
        ----------
        req_id : str
            Unique requirement identifier (e.g., "REQ-001").
        label : str
            Human-readable requirement name.
        description : str
            Detailed requirement description.
        related_features : list of str
            Feature IDs related to this requirement (e.g., "FEAT_HOLE", "PRIM_CYLINDER").
        related_ops : list of str
            Operation IDs related to this requirement (e.g., "OP-001", "OP-002").
        """
        node = GraphNode(
            id=req_id,
            label=label,
            node_type="requirement",
            attributes={
                "description": description,
                "related_features": related_features or [],
                "related_ops": related_ops or [],
            },
        )
        if req_id in self._nodes:
            raise ValueError(f"node {req_id!r} already exists")
        self._nodes[req_id] = node
        self._out.setdefault(req_id, [])
        self._in.setdefault(req_id, [])
        return node

    def find_requirements_by_feature(self, feature_id: str) -> list[GraphNode]:
        """Find all requirement nodes related to a given feature ID."""
        matching = []
        for node in self._nodes.values():
            attrs = node.attributes
            if "related_features" in attrs and feature_id in attrs["related_features"]:
                matching.append(node)
        return matching

    def find_requirements_by_op(self, op_id: str) -> list[GraphNode]:
        """Find all requirement nodes related to a given operation ID."""
        matching = []
        for node in self._nodes.values():
            attrs = node.attributes
            if "related_ops" in attrs and op_id in attrs["related_ops"]:
                matching.append(node)
        return matching

    def requirement_traceability_path(
        self,
        from_req: str,
        to_feature: str | None = None,
        to_op: str | None = None,
    ) -> list[dict[str, Any]]:
        """Trace from a requirement to related features or operations.

        Returns a list of dicts with 'node', 'edge', 'path' information.
        """
        path = []
        if from_req not in self._nodes:
            return path

        visited = {from_req}
        current = from_req
        while True:
            node = self._nodes.get(current)
            if node is None:
                break

            # Check if this node connects to the target
            if to_feature and "related_features" in node.attributes:
                if to_feature in node.attributes["related_features"]:
                    path.append(
                        {
                            "node": current,
                            "label": node.label,
                            "attributes": dict(node.attributes),
                        }
                    )
                    break
            if to_op and "related_ops" in node.attributes:
                if to_op in node.attributes["related_ops"]:
                    path.append(
                        {
                            "node": current,
                            "label": node.label,
                            "attributes": dict(node.attributes),
                        }
                    )
                    break

            # Follow outgoing edges
            next_nodes = []
            for edge in self._out.get(current, []):
                if edge.target not in visited:
                    next_nodes.append(edge.target)
                    visited.add(edge.target)

            if not next_nodes:
                break

            # Pick the first unvisited node (simple traversal)
            current = next_nodes[0]
            path.append(
                {
                    "node": current,
                    "label": self._nodes[current].label,
                    "attributes": dict(self._nodes[current].attributes),
                    "edge": edge.relation,
                }
            )

        return path

    # -------------------------------------------------------------- exports

    __all__ = [
        "GraphNode",
        "GraphEdge",
        "KnowledgeGraph",
        "add_requirement",
        "find_requirements_by_feature",
        "find_requirements_by_op",
        "requirement_traceability_path",
    ]