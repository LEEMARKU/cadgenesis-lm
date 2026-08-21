"""cadgenesis.execution.topology_analysis
========================================
Topology analysis for the CAD execution pipeline.

Adjacency graphs, connected components, manifold/closed/genus classification
and edge-usage auditing for triangle meshes, B-Rep topology graphs and raw
face meshes — composed over the existing substrate
(`cad.modeling.brep.TopologyGraph`, `cad.mesh.Mesh`, `reasoning.topology`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from cadgenesis.cad.mesh.mesh import Mesh
from cadgenesis.cad.modeling.brep import TopologyGraph
from cadgenesis.reasoning.topology import TopologyAnalyzer as _MeshTopologyAnalyzer


@dataclass
class TopologyCheck:
    """Single topology check result."""

    name: str
    passed: bool
    severity: str = "error"
    detail: str = ""
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "detail": self.detail,
            "recommendation": self.recommendation,
        }


@dataclass
class TopologyAnalysisReport:
    """Aggregated topology analysis result."""

    checks: list[TopologyCheck] = field(default_factory=list)
    graphs: dict[str, Any] = field(default_factory=dict)
    components: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed(self) -> list[TopologyCheck]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "total": len(self.checks),
            "failed": [c.name for c in self.checks if not c.passed],
            "graphs": self.graphs,
            "components": self.components,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "checks": [c.to_dict() for c in self.checks],
            "graphs": self.graphs,
            "components": self.components,
        }


class TopologyAnalyzer:
    """Execution-layer topology analyzer.

    Every method returns a :class:`TopologyAnalysisReport` carrying the
    analysis result plus the computed adjacency graphs and component counts.
    """

    # ------------------------------------------------------------------ mesh

    def analyze_mesh(self, mesh: Mesh) -> TopologyAnalysisReport:
        """Analyze a triangle mesh: edge usage, boundary, components."""
        report = TopologyAnalysisReport()
        if mesh.vertex_count == 0:
            report.checks.append(TopologyCheck("mesh:structure", False, detail="empty"))
            return report
        edges = mesh.undirected_edges()
        boundary = mesh.boundary_edges()
        watertight = mesh.is_watertight()
        report.graphs = {
            "edge_usage": {f"{a}:{b}": count for (a, b), count in sorted(edges.items())},
            "boundary_edges": [list(e) for e in sorted(boundary)],
        }
        report.checks.append(
            TopologyCheck(
                "mesh:edge_usage",
                all(c == 2 for c in edges.values()),
                detail=(
                    f"{len(boundary)} boundary edges of {len(edges)} total"
                    if boundary
                    else "every edge used exactly twice"
                ),
                recommendation="Fill boundary loops before export",
            )
        )
        report.checks.append(
            TopologyCheck(
                "mesh:closed",
                watertight,
                detail="closed" if watertight else "open",
            )
        )
        components = _mesh_components(mesh)
        report.components = {
            "face_components": len(components),
            "sizes": sorted((len(c) for c in components), reverse=True),
        }
        report.checks.append(
            TopologyCheck(
                "mesh:connected",
                len(components) == 1,
                severity="warning",
                detail=f"{len(components)} face components",
            )
        )
        return report

    def analyze_faces(self, faces: Sequence[Sequence[int]]) -> TopologyAnalysisReport:
        """Analyze raw face-index lists (reuses ``reasoning.topology``)."""
        report = TopologyAnalysisReport()
        if not faces:
            report.checks.append(TopologyCheck("faces:structure", False, detail="empty"))
            return report
        analyzer = _MeshTopologyAnalyzer()
        stats = analyzer.analyze_mesh(faces)
        report.checks.append(
            TopologyCheck(
                "faces:manifold",
                bool(stats.is_manifold),
                detail=f"{len(faces)} faces",
            )
        )
        report.checks.append(
            TopologyCheck(
                "faces:closed",
                bool(stats.is_closed),
                detail=f"genus {int(stats.genus)}",
            )
        )
        report.graphs = {"adjacency": analyzer.adjacency_graph(faces)}
        return report

    # ------------------------------------------------------------------ brep

    def analyze_brep(self, graph: TopologyGraph) -> TopologyAnalysisReport:
        """Analyze a B-Rep topology graph: graphs, components, Euler characteristic."""
        report = TopologyAnalysisReport()
        report.graphs = {
            "faces": {k: sorted(v) for k, v in graph.face_graph().adjacency.items()},
            "edges": {k: sorted(v) for k, v in graph.edge_graph().adjacency.items()},
        }
        components = graph.face_graph().connected_components()
        report.components = {"face_components": len(components)}
        checks = [
            TopologyCheck(
                "brep:manifold",
                graph.is_manifold(),
                detail=f"V={graph.vertex_count} E={graph.edge_count} F={graph.face_count}",
            ),
            TopologyCheck(
                "brep:closed",
                graph.is_closed(),
                detail=f"euler {graph.euler_characteristic()}, genus {graph.genus()}",
            ),
            TopologyCheck(
                "brep:connected",
                len(components) <= 1,
                severity="warning",
                detail=f"{len(components)} face components",
            ),
        ]
        report.checks.extend(checks)
        return report

    # ----------------------------------------------------------------- misc

    def summary(self) -> dict[str, Any]:
        return {"graphs": ("face", "edge", "vertex"), "mesh_analyzers": 1}


def _mesh_components(mesh: Mesh) -> list[list[int]]:
    """Connected components of the face adjacency graph (via shared edges)."""
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for index, face in enumerate(mesh.faces):
        pairs = {
            (min(face[0], face[1]), max(face[0], face[1])),
            (min(face[1], face[2]), max(face[1], face[2])),
            (min(face[2], face[0]), max(face[2], face[0])),
        }
        for pair in pairs:
            edge_faces.setdefault(pair, []).append(index)
    adjacency: dict[int, set[int]] = {i: set() for i in range(mesh.face_count)}
    for holders in edge_faces.values():
        if len(holders) == 2:
            adjacency[holders[0]].add(holders[1])
            adjacency[holders[1]].add(holders[0])
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in range(mesh.face_count):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


__all__ = [
    "TopologyAnalysisReport",
    "TopologyAnalyzer",
    "TopologyCheck",
]
