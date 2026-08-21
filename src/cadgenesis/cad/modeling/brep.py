"""cadgenesis.cad.modeling.brep
=============================
Boundary representation (B-Rep) solid model with full topology graphs.

The B-Rep stores vertices, edges, faces and shells as first-class objects
and exposes four adjacency graphs:

- :class:`TopologyGraph` — combined vertex/edge/face adjacency
- :class:`FaceGraph`       — faces sharing edges
- :class:`EdgeGraph`       — edges sharing vertices
- :class:`VertexGraph`     — vertices joined by edges

Topological sanity checks (manifold, closed shell, Euler-Poincare, genus)
are computed directly from the stored topology.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from cadgenesis.cad.geometry.core import Vec


@dataclass
class Vertex:
    """A 0-D topological entity: a point in space."""

    id: str
    point: Vec

    def __post_init__(self) -> None:
        if not isinstance(self.point, Vec):
            self.point = Vec.from_sequence(self.point)


@dataclass
class Edge:
    """A 1-D topological entity connecting two vertices."""

    id: str
    vertex_a: str
    vertex_b: str
    curve_type: str = "LINE"

    @property
    def endpoints(self) -> tuple[str, str]:
        return (self.vertex_a, self.vertex_b)


@dataclass
class Face:
    """A 2-D topological entity: a bounded surface region."""

    id: str
    surface_type: str = "PLANAR"
    vertices: list[str] = field(default_factory=list)
    edges: list[str] = field(default_factory=list)
    inner_loops: int = 0  # number of inner boundary loops (face holes)

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


@dataclass
class Shell:
    """A connected set of faces forming a closed (or open) boundary."""

    id: str
    face_ids: list[str] = field(default_factory=list)


class TopologyGraph:
    """Combined B-Rep adjacency graph.

    Edges of the graph are stored as ``(from_id, to_id, kind)`` where
    ``kind`` is one of ``"vertex"``, ``"edge"`` or ``"face"``.
    """

    def __init__(self) -> None:
        self.vertices: dict[str, Vertex] = {}
        self.edges: dict[str, Edge] = {}
        self.faces: dict[str, Face] = {}
        self.shells: dict[str, Shell] = {}

    def add_vertex(self, vertex: Vertex) -> Vertex:
        if vertex.id in self.vertices:
            raise KeyError(f"vertex {vertex.id!r} already exists")
        self.vertices[vertex.id] = vertex
        return vertex

    def add_edge(self, edge: Edge) -> Edge:
        if edge.id in self.edges:
            raise KeyError(f"edge {edge.id!r} already exists")
        if edge.vertex_a not in self.vertices or edge.vertex_b not in self.vertices:
            raise KeyError("edge references unknown vertices")
        self.edges[edge.id] = edge
        return edge

    def add_face(self, face: Face) -> Face:
        if face.id in self.faces:
            raise KeyError(f"face {face.id!r} already exists")
        unknown = [v for v in face.vertices if v not in self.vertices]
        if unknown:
            raise KeyError(f"face references unknown vertices {unknown}")
        self.faces[face.id] = face
        return face

    def add_shell(self, shell: Shell) -> Shell:
        if shell.id in self.shells:
            raise KeyError(f"shell {shell.id!r} already exists")
        unknown = [f for f in shell.face_ids if f not in self.faces]
        if unknown:
            raise KeyError(f"shell references unknown faces {unknown}")
        self.shells[shell.id] = shell
        return shell

    # -- graphs --------------------------------------------------------------
    def face_graph(self) -> FaceGraph:
        adjacency: dict[str, set[str]] = {f: set() for f in self.faces}
        edge_faces: dict[str, list[str]] = defaultdict(list)
        for face in self.faces.values():
            for edge_id in face.edges:
                edge_faces[edge_id].append(face.id)
        for faces in edge_faces.values():
            for a in faces:
                for b in faces:
                    if a != b:
                        adjacency[a].add(b)
        return FaceGraph(adjacency)

    def edge_graph(self) -> EdgeGraph:
        adjacency: dict[str, set[str]] = {e: set() for e in self.edges}
        for edge in self.edges.values():
            for other in self.edges.values():
                if edge.id != other.id and (
                    edge.vertex_a == other.vertex_a
                    or edge.vertex_a == other.vertex_b
                    or edge.vertex_b == other.vertex_a
                    or edge.vertex_b == other.vertex_b
                ):
                    adjacency[edge.id].add(other.id)
        return EdgeGraph(adjacency)

    def vertex_graph(self) -> VertexGraph:
        adjacency: dict[str, set[str]] = {v: set() for v in self.vertices}
        for edge in self.edges.values():
            adjacency[edge.vertex_a].add(edge.vertex_b)
            adjacency[edge.vertex_b].add(edge.vertex_a)
        return VertexGraph(adjacency)

    def topology_graph(self) -> TopologyGraph:
        return self

    # -- counts --------------------------------------------------------------
    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def shell_count(self) -> int:
        return len(self.shells)

    def euler_characteristic(self) -> int:
        return self.vertex_count - self.edge_count + self.face_count

    def edge_face_incidence(self) -> dict[str, int]:
        """Count how many faces reference each edge."""
        counts: dict[str, int] = {}
        for face in self.faces.values():
            for edge_id in face.edges:
                counts[edge_id] = counts.get(edge_id, 0) + 1
        return counts

    # -- topology analysis ----------------------------------------------------
    def is_manifold(self) -> bool:
        """True when every edge is incident to at most two faces."""
        return all(count <= 2 for count in self.edge_face_incidence().values())

    def is_closed(self) -> bool:
        """True when every edge is incident to exactly two faces."""
        incidence = self.edge_face_incidence()
        return bool(incidence) and all(count == 2 for count in incidence.values())

    def genus(self) -> int:
        """Through-hole count from the Euler-Poincare relation.

        For a closed manifold solid: ``chi = 2*S + L - 2*G`` where ``S`` is
        the number of shells and ``L`` the number of *inner* face loops.
        """
        if not self.is_closed():
            return 0
        chi = self.euler_characteristic()
        shells = self.shell_count
        inner_loops = sum(face.inner_loops for face in self.faces.values())
        numerator = 2 * shells + inner_loops - chi
        if numerator < 0 or numerator % 2 != 0:
            return 0
        return numerator // 2

    def analyze(self) -> dict[str, Any]:
        return {
            "vertices": self.vertex_count,
            "edges": self.edge_count,
            "faces": self.face_count,
            "shells": self.shell_count,
            "euler_characteristic": self.euler_characteristic(),
            "genus": self.genus(),
            "is_manifold": self.is_manifold(),
            "is_closed": self.is_closed(),
        }

    def validate(self) -> list[str]:
        """Topological validation: returns a list of problems (empty = valid)."""
        problems = [
            f"edge {edge.id!r} is degenerate (zero length)"
            for edge in self.edges.values()
            if edge.vertex_a == edge.vertex_b
        ]
        problems.extend(
            f"face {face.id!r} has fewer than 3 vertices"
            for face in self.faces.values()
            if face.vertex_count < 3
        )
        self.edge_face_incidence()
        if not self.is_manifold():
            problems.append("model is non-manifold (an edge is shared by >2 faces)")
        chi = self.euler_characteristic()
        shells = self.shell_count
        inner_loops = sum(face.inner_loops for face in self.faces.values())
        numerator = 2 * shells + inner_loops - chi
        if numerator < 0 or numerator % 2 != 0:
            problems.append("Euler-Poincare consistency check failed")
        return problems


class GraphBase:
    """Base class for the three specialised adjacency graphs."""

    def __init__(self, adjacency: dict[str, set[str]]) -> None:
        self.adjacency = adjacency

    def neighbors(self, node: str) -> list[str]:
        return sorted(self.adjacency.get(node, set()))

    def nodes(self) -> list[str]:
        return list(self.adjacency)

    def degree(self, node: str) -> int:
        return len(self.adjacency.get(node, set()))

    def degrees(self) -> dict[str, int]:
        return {node: len(neighbors) for node, neighbors in self.adjacency.items()}

    def connected_components(self) -> list[list[str]]:
        seen: set[str] = set()
        components: list[list[str]] = []
        for start in self.adjacency:
            if start in seen:
                continue
            stack = [start]
            component: list[str] = []
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                component.append(node)
                stack.extend(self.adjacency.get(node, set()))
            components.append(component)
        return components

    def to_dict(self) -> dict[str, list[str]]:
        return {node: sorted(neighbors) for node, neighbors in self.adjacency.items()}


class FaceGraph(GraphBase):
    """Faces as nodes; an edge connects two faces sharing a B-Rep edge."""


class EdgeGraph(GraphBase):
    """Edges as nodes; an edge connects two B-Rep edges sharing a vertex."""


class VertexGraph(GraphBase):
    """Vertices as nodes; an edge connects two vertices joined by a B-Rep edge."""


class BRepSolid:
    """A solid as a boundary representation with topology graphs.

    Use the factory helpers (:meth:`from_prism`, :meth:`from_faces`) or build
    the topology manually with :class:`TopologyGraph`.
    """

    def __init__(self, graph: TopologyGraph) -> None:
        self.graph = graph

    @classmethod
    def from_prism(
        cls,
        length: float = 10.0,
        width: float = 5.0,
        height: float = 3.0,
        name: str = "prism",
    ) -> BRepSolid:
        """Build a closed hexahedral B-Rep solid from a box dimension."""
        lx, ly, lz = length / 2, width / 2, height / 2
        points = {
            "v0": Vec(-lx, -ly, -lz),
            "v1": Vec(lx, -ly, -lz),
            "v2": Vec(lx, ly, -lz),
            "v3": Vec(-lx, ly, -lz),
            "v4": Vec(-lx, -ly, lz),
            "v5": Vec(lx, -ly, lz),
            "v6": Vec(lx, ly, lz),
            "v7": Vec(-lx, ly, lz),
        }
        graph = TopologyGraph()
        for vid, point in points.items():
            graph.add_vertex(Vertex(vid, point))
        # 12 edges
        edge_specs = [
            ("e0", "v0", "v1"),
            ("e1", "v1", "v2"),
            ("e2", "v2", "v3"),
            ("e3", "v3", "v0"),
            ("e4", "v4", "v5"),
            ("e5", "v5", "v6"),
            ("e6", "v6", "v7"),
            ("e7", "v7", "v4"),
            ("e8", "v0", "v4"),
            ("e9", "v1", "v5"),
            ("e10", "v2", "v6"),
            ("e11", "v3", "v7"),
        ]
        for eid, va, vb in edge_specs:
            graph.add_edge(Edge(eid, va, vb))
        # 6 faces (bottom, top, front, back, left, right) — edge loops follow
        # the face vertex order
        face_specs = [
            ("f_bottom", ["e0", "e1", "e2", "e3"]),
            ("f_top", ["e4", "e5", "e6", "e7"]),
            ("f_front", ["e0", "e9", "e4", "e8"]),
            ("f_back", ["e2", "e11", "e6", "e10"]),
            ("f_left", ["e3", "e11", "e7", "e8"]),
            ("f_right", ["e1", "e10", "e5", "e9"]),
        ]
        face_vertices = {
            "f_bottom": ["v0", "v1", "v2", "v3"],
            "f_top": ["v4", "v5", "v6", "v7"],
            "f_front": ["v0", "v1", "v5", "v4"],
            "f_back": ["v2", "v3", "v7", "v6"],
            "f_left": ["v0", "v3", "v7", "v4"],
            "f_right": ["v1", "v2", "v6", "v5"],
        }
        for fid, edge_ids in face_specs:
            graph.add_face(Face(fid, vertices=face_vertices[fid], edges=edge_ids))
        graph.add_shell(Shell("shell0", [f for f, _ in face_specs]))
        return cls(graph)

    @classmethod
    def from_faces(cls, faces: list[list[str]], vertex_points: dict[str, Vec]) -> BRepSolid:
        """Build a B-Rep from faces (each a list of vertex ids) and a point map.

        Edges are derived automatically; faces may be triangular or polygonal.
        """
        graph = TopologyGraph()
        for vid, point in vertex_points.items():
            graph.add_vertex(Vertex(vid, point))
        edge_lookup: dict[tuple[str, str], str] = {}
        edge_counter = 0
        for face_index, vertex_ids in enumerate(faces):
            face_edges: list[str] = []
            for i, va in enumerate(vertex_ids):
                vb = vertex_ids[(i + 1) % len(vertex_ids)]
                key = (va, vb) if va < vb else (vb, va)
                if key not in edge_lookup:
                    edge_lookup[key] = f"e{edge_counter}"
                    edge_counter += 1
                    graph.add_edge(Edge(edge_lookup[key], key[0], key[1]))
                face_edges.append(edge_lookup[key])
            graph.add_face(Face(f"f{face_index}", vertices=list(vertex_ids), edges=face_edges))
        return cls(graph)

    # -- delegation -----------------------------------------------------------
    @property
    def topology_graph(self) -> TopologyGraph:
        return self.graph

    def face_graph(self) -> FaceGraph:
        return self.graph.face_graph()

    def edge_graph(self) -> EdgeGraph:
        return self.graph.edge_graph()

    def vertex_graph(self) -> VertexGraph:
        return self.graph.vertex_graph()

    def validate(self) -> list[str]:
        return self.graph.validate()

    def analyze(self) -> dict[str, Any]:
        return self.graph.analyze()

    def volume(self) -> float:
        """Analytic volume (box prism only, extensible)."""
        points = self.graph.vertices
        if len(points) == 8:
            v0 = points["v0"].point
            v1 = points["v1"].point
            v3 = points["v3"].point
            v4 = points["v4"].point
            return abs((v1 - v0).dot((v3 - v0).cross(v4 - v0)))
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "vertices": {vid: v.point.to_list() for vid, v in self.graph.vertices.items()},
            "faces": [f.vertices for f in self.graph.faces.values()],
            "analysis": self.analyze(),
        }


__all__ = [
    "BRepSolid",
    "Edge",
    "EdgeGraph",
    "Face",
    "FaceGraph",
    "GraphBase",
    "Shell",
    "TopologyGraph",
    "Vertex",
    "VertexGraph",
]
