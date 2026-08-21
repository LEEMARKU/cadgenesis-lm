"""cadgenesis.cad.mechanisms.joints
================================
Mechanism joints and degree-of-freedom analysis.

Each joint class has a spatial DOF count (revolute=1, prismatic=1,
cylindrical=2, spherical=3, planar=3, universal=2, screw=1, welded=0).
The Gruebler-Kutzbach formula computes the overall mechanism mobility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

JOINT_TYPES = (
    "REVOLUTE",
    "PRISMATIC",
    "CYLINDRICAL",
    "SPHERICAL",
    "PLANAR",
    "UNIVERSAL",
    "SCREW",
    "WELDED",
)

_JOINT_DOF: dict[str, int] = {
    "REVOLUTE": 1,
    "PRISMATIC": 1,
    "CYLINDRICAL": 2,
    "SPHERICAL": 3,
    "PLANAR": 3,
    "UNIVERSAL": 2,
    "SCREW": 1,
    "WELDED": 0,
}


@dataclass
class Joint:
    """A kinematic joint connecting two links at a point."""

    name: str
    joint_type: str
    link_a: str
    link_b: str
    position: Any = None  # optional Vec

    def __post_init__(self) -> None:
        if self.joint_type not in JOINT_TYPES:
            raise ValueError(
                f"unknown joint type {self.joint_type!r}; expected one of {JOINT_TYPES}"
            )

    @property
    def dof(self) -> int:
        return _JOINT_DOF[self.joint_type]

    @property
    def constraints_removed(self) -> int:
        # spatial mobility of a rigid body is 6
        return 6 - self.dof

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "joint_type": self.joint_type,
            "link_a": self.link_a,
            "link_b": self.link_b,
        }


@dataclass
class Mechanism:
    """A collection of links and joints with mobility analysis."""

    name: str
    links: list[str] = field(default_factory=list)
    joints: list[Joint] = field(default_factory=list)

    def add_link(self, name: str) -> str:
        if name in self.links:
            raise KeyError(f"link {name!r} already exists")
        self.links.append(name)
        return name

    def add_joint(self, joint: Joint) -> Joint:
        if joint.link_a not in self.links or joint.link_b not in self.links:
            raise KeyError("joint references an unknown link")
        self.joints.append(joint)
        return joint

    @property
    def link_count(self) -> int:
        return len(self.links)

    @property
    def joint_count(self) -> int:
        return len(self.joints)

    def mobility_planar(self, grounded_links: int = 1) -> int:
        """Gruebler-Kutzbach for planar mechanisms:
        DOF = 3*(n - 1) + sum(joint_dof - 3) for each joint.
        """
        return 3 * (self.link_count - 1) + sum(j.dof - 3 for j in self.joints)

    def mobility_spatial(self, grounded_links: int = 1) -> int:
        """Gruebler-Kutzbach for spatial mechanisms:
        DOF = 6*(n - 1) + sum(joint dof - 6) for each joint.
        """
        return 6 * (self.link_count - 1) + sum(j.dof - 6 for j in self.joints)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "links": list(self.links),
            "joints": [j.to_dict() for j in self.joints],
            "mobility_planar": self.mobility_planar(),
            "mobility_spatial": self.mobility_spatial(),
        }


__all__ = ["JOINT_TYPES", "Joint", "Mechanism"]
