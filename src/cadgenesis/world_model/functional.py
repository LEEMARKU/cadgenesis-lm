"""cadgenesis.world_model.functional
===================================
Functional reasoning (Pillar 4).

The :class:`FunctionalReasoner` verifies that the world model satisfies
*functional requirements*: prescribed degrees of freedom, load paths,
reachability, flow continuity and top-level capabilities.  It operates over
:class:`~cadgenesis.world_model.objects.WorldObject` graphs and the
assemblies produced by :mod:`cadgenesis.world_model.assembly`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cadgenesis.world_model.objects import WorldObject


@dataclass
class FunctionalCheck:
    """Result of a single functional verification."""

    name: str
    passed: bool
    details: str = ""
    values: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
            "values": dict(self.values),
        }


class FunctionalReasoner:
    """Functional verification over world-model objects."""

    # ------------------------------------------------------------------ dof

    def available_dof(self, obj: WorldObject) -> int:
        """Number of free degrees of freedom of an object.

        Six minus the number of constraints reported by the parent/child
        relations in the world model.  Uses the ``constraints`` relation on
        the object graph when present.
        """
        constrained = obj.relations.get("constraints", [])
        return max(0, 6 - len(constrained))

    def requires_dof(
        self,
        obj: WorldObject,
        minimum: int,
    ) -> FunctionalCheck:
        """Check that an object keeps at least ``minimum`` free DOF."""
        dof = self.available_dof(obj)
        passed = dof >= minimum
        return FunctionalCheck(
            name=f"dof.{obj.name}",
            passed=passed,
            details=f"free DOF {dof} (required >= {minimum})",
            values={"dof": dof, "required": minimum},
        )

    # ------------------------------------------------------------- fit box

    def fits_in_box(
        self,
        obj: WorldObject,
        width: float,
        height: float,
        depth: float,
    ) -> FunctionalCheck:
        """Check local bounding box fits inside a rectangular envelope."""
        lo, hi = obj.bounds()
        ex = hi.x - lo.x
        ey = hi.y - lo.y
        ez = hi.z - lo.z
        passed = ex <= width + 1e-9 and ey <= height + 1e-9 and ez <= depth + 1e-9
        return FunctionalCheck(
            name=f"envelope.{obj.name}",
            passed=passed,
            details=f"extents ({ex:.2f},{ey:.2f},{ez:.2f}) vs ({width},{height},{depth})",
            values={"extents": (ex, ey, ez), "envelope": (width, height, depth)},
        )

    # ------------------------------------------------------------ load path

    def load_path(
        self,
        source: WorldObject,
        sink: WorldObject,
        graph: Any,
    ) -> FunctionalCheck:
        """Check that a structural load path exists between two objects.

        The path is found over the object graph's ``supports`` / ``mounted_on``
        relations via a simple BFS on ``children`` + ``parents``.
        """
        nodes = [graph.root_for(source), graph.root_for(sink)] if hasattr(graph, "root_for") else []
        if not nodes or nodes[0] is None or nodes[1] is None:
            return FunctionalCheck(
                name=f"load_path.{source.name}.{sink.name}",
                passed=False,
                details="objects not placed in graph",
            )
        visited: set[str] = set()
        queue = [source.object_id]
        found = False
        while queue and not found:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if current == sink.object_id:
                found = True
                break
            for neighbor in graph.neighbors(current):
                queue = [*queue, neighbor]
        return FunctionalCheck(
            name=f"load_path.{source.name}.{sink.name}",
            passed=found,
            details=("structural connection exists" if found else "no load path found"),
            values={"connected": found},
        )

    # -------------------------------------------------------------- flow

    def flow_continuity(
        self,
        path: list[WorldObject],
    ) -> FunctionalCheck:
        """Check every consecutive pair in a flow path is in contact.

        Contact is inferred from the ``supports`` / ``mounted_on`` relations
        between the objects.
        """
        gaps: list[tuple[str, str]] = []
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            a_children = set(a.relations.get("children", []))
            b_children = set(b.relations.get("children", []))
            connected = b.object_id in a_children or a.object_id in b_children
            connected = connected or b.object_id in b_children or a.object_id in a_children
            if not connected:
                gaps.append((a.name, b.name))
        passed = not gaps
        return FunctionalCheck(
            name="flow_continuity",
            passed=passed,
            details=("flow path continuous" if passed else f"disconnected at {gaps}"),
            values={"gaps": gaps},
        )


__all__ = ["FunctionalCheck", "FunctionalReasoner"]
