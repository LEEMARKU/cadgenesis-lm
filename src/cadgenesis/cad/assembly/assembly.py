"""cadgenesis.cad.assembly.assembly
===============================
Assembly modelling: hierarchical assemblies of components (parts or
sub-assemblies) with placement transforms and parent references.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cadgenesis.cad.geometry.core import Transform


@dataclass
class Component:
    """A node in the assembly tree (a part instance or a sub-assembly)."""

    name: str
    is_assembly: bool = False
    children: list[Component] = field(default_factory=list)
    transform: Transform = field(default_factory=Transform.identity)
    part_id: str = ""  # reference to a part model when not an assembly

    def add_child(self, child: Component) -> Component:
        if not self.is_assembly:
            raise ValueError(f"component {self.name!r} is a part, not an assembly")
        self.children.append(child)
        return child

    @property
    def is_part(self) -> bool:
        return not self.is_assembly

    def walk(self):
        """Depth-first traversal of the component tree (self first)."""
        yield self
        for child in self.children:
            yield from child.walk()

    def descendant_count(self) -> int:
        return sum(1 for _ in self.walk()) - 1

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "is_assembly": self.is_assembly,
            "part_id": self.part_id,
            "transform": self.transform.to_list(),
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Component:
        component = cls(
            name=str(data["name"]),
            is_assembly=bool(data.get("is_assembly", False)),
            part_id=str(data.get("part_id", "")),
            transform=(
                Transform(data.get("transform")) if data.get("transform") else Transform.identity()
            ),
        )
        for child_data in data.get("children", []):
            component.add_child(cls.from_dict(child_data))
        return component


class Assembly:
    """A hierarchical assembly document.

    ``root`` is the top-level component (an assembly); the tree holds all
    instances.  Placement of any component is its local transform composed
    with its parent's (accumulated) transform.
    """

    def __init__(self, name: str = "assembly", root: Component | None = None) -> None:
        self.name = name
        self.root = root if root is not None else Component(name, is_assembly=True)

    def add_part(
        self,
        name: str,
        part_id: str,
        parent: Component | None = None,
        transform: Transform | None = None,
    ) -> Component:
        part = Component(
            name, is_assembly=False, part_id=part_id, transform=transform or Transform.identity()
        )
        (parent or self.root).add_child(part)
        return part

    def add_subassembly(
        self, name: str, parent: Component | None = None, transform: Transform | None = None
    ) -> Component:
        sub = Component(name, is_assembly=True, transform=transform or Transform.identity())
        (parent or self.root).add_child(sub)
        return sub

    def parts(self) -> list[Component]:
        return [c for c in self.root.walk() if c.is_part]

    def components(self) -> list[Component]:
        return list(self.root.walk())

    def find(self, name: str) -> Component | None:
        for component in self.root.walk():
            if component.name == name:
                return component
        return None

    def world_transform(self, name: str) -> Transform:
        """Accumulated transform of a component from the assembly root."""
        path = self._path_to(name)
        if path is None:
            raise KeyError(f"component {name!r} not found in assembly")
        result = Transform.identity()
        for component in path:
            result = result.composed(component.transform)
        return result

    def _path_to(self, name: str) -> list[Component] | None:
        def search(component: Component, path: list[Component]) -> list[Component] | None:
            path.append(component)
            if component.name == name:
                return list(path)
            for child in component.children:
                found = search(child, path)
                if found is not None:
                    return found
            path.pop()
            return None

        return search(self.root, [])

    def part_count(self) -> int:
        return len(self.parts())

    def unique_part_ids(self) -> set[str]:
        return {p.part_id for p in self.parts() if p.part_id}

    def max_depth(self) -> int:
        return self.root.depth()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "root": self.root.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Assembly:
        return cls(name=str(data.get("name", "assembly")), root=Component.from_dict(data["root"]))


__all__ = ["Assembly", "Component"]
