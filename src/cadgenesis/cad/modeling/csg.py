"""cadgenesis.cad.modeling.csg
===========================
Constructive solid geometry: a binary tree of boolean operations over solid
primitives, with operation history tracking.

CSG is a complementary representation to B-Rep: instead of boundary faces the
model is described as a tree of *union / subtraction / intersection* over
primitive volumes.  Each node records its own creation event so the entire
construction history is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from cadgenesis.cad.geometry.core import Transform, Vec
from cadgenesis.cad.modeling.primitives import SolidPrimitive

OPERATIONS = ("UNION", "SUBTRACT", "INTERSECT")


@dataclass
class CSGNode:
    """A node in the CSG tree.

    ``op`` is ``None`` for a primitive leaf; otherwise one of
    ``UNION`` / ``SUBTRACT`` / ``INTERSECT`` with exactly two children.
    """

    id: str
    op: str | None = None
    left: CSGNode | None = None
    right: CSGNode | None = None
    primitive: SolidPrimitive | None = None
    transform: Transform | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if self.op is not None:
            if self.op not in OPERATIONS:
                raise ValueError(f"unknown CSG operation {self.op!r}")
            if self.left is None or self.right is None:
                raise ValueError("a binary CSG node needs both children")
        elif self.primitive is None:
            raise ValueError("a leaf CSG node needs a primitive")

    @property
    def is_leaf(self) -> bool:
        return self.op is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "op": self.op,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
            "primitive": self.primitive.to_dict() if self.primitive else None,
            "transform": self.transform.to_list() if self.transform else None,
            "created_at": self.created_at,
        }


@dataclass
class CSGOperation:
    """A recorded CSG operation (for history tracking)."""

    op: str
    node_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "note": self.note,
        }


class CSGTree:
    """A CSG tree with a root node and an append-only operation history."""

    def __init__(self, root: CSGNode | None = None) -> None:
        self.root = root
        self._nodes: dict[str, CSGNode] = {}
        self.history: list[CSGOperation] = []
        self._counter = 0
        if root is not None:
            self._register(root)

    def _register(self, node: CSGNode) -> None:
        if node.id in self._nodes:
            return  # already registered (idempotent)
        self._nodes[node.id] = node
        if node.left is not None:
            self._register(node.left)
        if node.right is not None:
            self._register(node.right)

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def new_leaf(
        self, primitive: SolidPrimitive, transform: Transform | None = None, node_id: str = ""
    ) -> CSGNode:
        node = CSGNode(node_id or self._next_id("leaf_"), primitive=primitive, transform=transform)
        self._register(node)
        return node

    def new_binary(self, op: str, left: CSGNode, right: CSGNode, node_id: str = "") -> CSGNode:
        if op not in OPERATIONS:
            raise ValueError(f"unknown CSG operation {op!r}")
        node = CSGNode(node_id or self._next_id("node_"), op=op, left=left, right=right)
        self._register(node)
        self.history.append(CSGOperation(op, node.id, note=f"{op}({left.id}, {right.id})"))
        return node

    # -- convenience operations -------------------------------------------------
    def union(self, a: CSGNode, b: CSGNode, node_id: str = "") -> CSGNode:
        return self.new_binary("UNION", a, b, node_id)

    def subtract(self, a: CSGNode, b: CSGNode, node_id: str = "") -> CSGNode:
        return self.new_binary("SUBTRACT", a, b, node_id)

    def intersect(self, a: CSGNode, b: CSGNode, node_id: str = "") -> CSGNode:
        return self.new_binary("INTERSECT", a, b, node_id)

    def set_root(self, node: CSGNode) -> None:
        self.root = node

    # -- queries ---------------------------------------------------------------
    @property
    def nodes(self) -> list[CSGNode]:
        return list(self._nodes.values())

    def node(self, node_id: str) -> CSGNode:
        if node_id not in self._nodes:
            raise KeyError(f"unknown CSG node {node_id!r}")
        return self._nodes[node_id]

    def leaves(self) -> list[CSGNode]:
        return [n for n in self._nodes.values() if n.is_leaf]

    def depth(self) -> int:
        return self._depth(self.root) if self.root else 0

    @staticmethod
    def _depth(node: CSGNode) -> int:
        if node.left is None or node.right is None:
            return 1
        return 1 + max(CSGTree._depth(node.left), CSGTree._depth(node.right))

    def bounds(self) -> tuple[Vec, Vec] | None:
        """Union AABB of all primitive leaves (a coarse solid bound)."""
        if self.root is None:
            return None
        mins = [float("inf")] * 3
        maxs = [float("-inf")] * 3
        for leaf in self.leaves():
            if leaf.primitive is None:
                continue
            t = leaf.transform
            lo, hi = leaf.primitive.aabb()
            corners = [
                Vec(lo.x, lo.y, lo.z),
                Vec(hi.x, lo.y, lo.z),
                Vec(lo.x, hi.y, lo.z),
                Vec(hi.x, hi.y, lo.z),
                Vec(lo.x, lo.y, hi.z),
                Vec(hi.x, lo.y, hi.z),
                Vec(lo.x, hi.y, hi.z),
                Vec(hi.x, hi.y, hi.z),
            ]
            points = t.apply_many(corners) if t else corners
            for p in points:
                for i in range(3):
                    mins[i] = min(mins[i], p[i])
                    maxs[i] = max(maxs[i], p[i])
        return Vec(*mins), Vec(*maxs)

    def validate(self) -> list[str]:
        """Structural validation of the CSG tree."""
        problems: list[str] = []
        if self.root is None:
            return ["CSG tree has no root"]
        if self.root.id not in self._nodes:
            problems.append("root node is not registered")
        for node in self._nodes.values():
            if node.op is None and node.primitive is None:
                problems.append(f"leaf node {node.id!r} has no primitive")
            if node.op is not None:
                if node.left is None or node.right is None:
                    problems.append(f"binary node {node.id!r} is missing a child")
                elif node.left.id not in self._nodes or node.right.id not in self._nodes:
                    problems.append(f"node {node.id!r} references unregistered children")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root.to_dict() if self.root else None,
            "history": [h.to_dict() for h in self.history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CSGTree:
        tree = cls()
        root_data = data.get("root")
        if root_data:
            tree.root = cls._node_from_dict(root_data, tree)
            tree._register(tree.root)
        for h in data.get("history", []):
            tree.history.append(
                CSGOperation(
                    str(h["op"]),
                    str(h["node_id"]),
                    str(h.get("timestamp", "")),
                    str(h.get("note", "")),
                )
            )
        return tree

    @classmethod
    def _node_from_dict(cls, data: dict[str, Any], tree: CSGTree) -> CSGNode:
        primitive_data = data.get("primitive")
        primitive = SolidPrimitive.from_dict(primitive_data) if primitive_data else None
        transform = Transform(data["transform"]) if data.get("transform") else None
        left = cls._node_from_dict(data["left"], tree) if data.get("left") else None
        right = cls._node_from_dict(data["right"], tree) if data.get("right") else None
        return CSGNode(
            id=str(data["id"]),
            op=data.get("op"),
            left=left,
            right=right,
            primitive=primitive,
            transform=transform,
            created_at=str(data.get("created_at", "")),
        )


__all__ = ["OPERATIONS", "CSGNode", "CSGOperation", "CSGTree"]
