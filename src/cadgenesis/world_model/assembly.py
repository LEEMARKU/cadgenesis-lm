"""cadgenesis.world_model.assembly
=================================
Assembly validation (Pillar 4).

:class:`AssemblyValidator` validates a mechanical assembly described by
parts, mates (reusing ``cadgenesis.cad.assembly.mates`` mate vocabulary) and
joints (reusing ``cadgenesis.cad.mechanisms.joints``).  Checks include part
reference integrity, mate/joint vocabulary validity, cycle detection and
mobility (Gruebler/Kutzbach style).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cadgenesis.cad.assembly.mates import MATE_TYPES
from cadgenesis.cad.mechanisms.joints import JOINT_TYPES, Joint
from cadgenesis.world_model.objects import WorldObject


@dataclass
class WorldAssembly:
    """A lightweight assembly description for validation."""

    name: str
    parts: list[WorldObject]
    mates: list[dict[str, Any]] = field(default_factory=list)
    joints: list[Joint] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def part_ids(self) -> set[str]:
        return {p.object_id for p in self.parts}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parts": [p.to_dict() for p in self.parts],
            "mates": list(self.mates),
            "joints": [j.to_dict() for j in self.joints],
            "metadata": dict(self.metadata),
        }


@dataclass
class AssemblyCheck:
    """Single assembly-validation result."""

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


class AssemblyValidator:
    """Validate part/mate/joint integrity and overall mobility."""

    def validate(self, assembly: WorldAssembly) -> list[AssemblyCheck]:
        checks: list[AssemblyCheck] = []
        part_ids = assembly.part_ids()

        if not assembly.parts:
            checks.append(AssemblyCheck("parts", False, "assembly has no parts"))
            return checks

        # -- mate vocabulary -----------------------------------------------
        bad_mates = [m["type"] for m in assembly.mates if m.get("type") not in MATE_TYPES]
        checks.append(
            AssemblyCheck(
                "mate_types",
                not bad_mates,
                "all mate types valid" if not bad_mates else f"unknown mate types: {bad_mates}",
            )
        )

        # -- joint vocabulary ----------------------------------------------
        bad_joints = [j.joint_type for j in assembly.joints if j.joint_type not in JOINT_TYPES]
        checks.append(
            AssemblyCheck(
                "joint_types",
                not bad_joints,
                "all joint types valid" if not bad_joints else f"unknown joint types: {bad_joints}",
            )
        )

        # -- mate references exist ------------------------------------------
        missing = []
        for mate in assembly.mates:
            for key in ("part_a", "part_b", "entity_a", "entity_b"):
                ref = mate.get(key)
                if (
                    isinstance(ref, str)
                    and ref not in part_ids
                    and ref not in {p.name for p in assembly.parts}
                ):
                    missing.append(ref)
        checks.append(
            AssemblyCheck(
                "mate_references",
                not missing,
                "all mate references resolve" if not missing else f"dangling refs: {missing}",
            )
        )

        # -- joint references exist -----------------------------------------
        def _resolve(ref: str) -> str:
            if ref in part_ids:
                return ref
            matched = next((p.object_id for p in assembly.parts if p.name == ref), ref)
            return matched

        dangling = [
            j.name
            for j in assembly.joints
            if _resolve(j.link_a) not in part_ids or _resolve(j.link_b) not in part_ids
        ]
        checks.append(
            AssemblyCheck(
                "joint_references",
                not dangling,
                "all joint links resolve" if not dangling else f"dangling joints: {dangling}",
            )
        )

        # -- connectivity graph ---------------------------------------------
        adjacency: dict[str, set[str]] = {p.object_id: set() for p in assembly.parts}
        for mate in assembly.mates:
            pa = mate.get("part_a")
            pb = mate.get("part_b")
            if isinstance(pa, str) and isinstance(pb, str):
                id_a = next(
                    (p.object_id for p in assembly.parts if p.object_id == pa or p.name == pa),
                    None,
                )
                id_b = next(
                    (p.object_id for p in assembly.parts if p.object_id == pb or p.name == pb),
                    None,
                )
                if id_a is not None and id_b is not None:
                    adjacency[id_a].add(id_b)
                    adjacency[id_b].add(id_a)
        for joint in assembly.joints:
            link_a = _resolve(joint.link_a)
            link_b = _resolve(joint.link_b)
            if link_a in adjacency and link_b in adjacency:
                adjacency[link_a].add(link_b)
                adjacency[link_b].add(link_a)

        # connectedness via BFS from first part
        root = assembly.parts[0].object_id
        seen: set[str] = set()
        queue = [root]
        while queue:
            node = queue.pop(0)
            if node in seen:
                continue
            seen.add(node)
            queue.extend(adjacency.get(node, set()) - seen)
        unreachable = part_ids - seen
        checks.append(
            AssemblyCheck(
                "connectivity",
                not unreachable,
                "assembly fully connected"
                if not unreachable
                else f"disconnected parts: {unreachable}",
            )
        )

        # -- cycle detection (simplified: more edges than tree for its size)
        edge_count = sum(len(n) for n in adjacency.values()) // 2
        has_cycle = edge_count >= len(part_ids)
        checks.append(
            AssemblyCheck(
                "cycle_free",
                not has_cycle,
                "no redundant cycles"
                if not has_cycle
                else f"{edge_count} edges vs {len(part_ids)} parts",
            )
        )

        # -- mobility (Gruebler/Kutzbach for spatial bodies) ----------------
        mobility = 6 * (len(part_ids) - 1) - sum(j.constraints_removed for j in assembly.joints)
        checks.append(
            AssemblyCheck(
                "mobility",
                mobility >= 0,
                f"computed mobility {mobility} (should be >= 0)",
                values={"mobility": mobility, "parts": len(part_ids)},
            )
        )

        return checks

    def validate_all(self, assembly: WorldAssembly) -> bool:
        """True only when every check passes."""
        return all(c.passed for c in self.validate(assembly))


__all__ = ["AssemblyCheck", "AssemblyValidator", "WorldAssembly"]
