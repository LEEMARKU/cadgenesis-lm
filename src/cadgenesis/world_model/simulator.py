"""cadgenesis.world_model.simulator
==================================
Motion simulation (Pillar 4).

:class:`MotionSimulator` provides a lightweight forward-kinematics simulator
for mechanisms described by ``cadgenesis.cad.mechanisms.joints`` plus path
collision checking that reuses :class:`cadgenesis.world_model.spatial.SpatialReasoner`.

The simulator resolves a serial kinematic chain: every joint contributes its
transform (rotation for revolute joints, translation for prismatic ones) and
the world transform of each link is the ordered composition along the chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cadgenesis.cad.geometry.core import Transform
from cadgenesis.cad.mechanisms.joints import Joint, Mechanism
from cadgenesis.world_model.objects import WorldObject
from cadgenesis.world_model.spatial import SpatialReasoner

_ROTATIONAL = ("REVOLUTE", "CYLINDRICAL", "SPHERICAL", "UNIVERSAL", "SCREW")
_TRANSLATIONAL = ("PRISMATIC", "SCREW")


@dataclass
class JointState:
    """A commanded value for one joint."""

    joint_name: str
    value: float  # radians for rotational, mm for prismatic


@dataclass
class SimulatedPose:
    """State of the mechanism at a moment in time."""

    time: float
    joint_values: dict[str, float]
    link_poses: dict[str, Transform]

    def position_of(self, link: str) -> tuple[float, float, float]:
        pose = self.link_poses[link]
        m = pose.matrix
        return (m[0][3], m[1][3], m[2][3])

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "joint_values": dict(self.joint_values),
            "link_poses": {link: pose.to_list() for link, pose in self.link_poses.items()},
        }


class MotionSimulator:
    """Forward-kinematics simulation of a serial mechanism."""

    def __init__(self, base: Transform | None = None) -> None:
        self.base = base if base is not None else Transform.identity()

    def _joint_transform(self, joint: Joint, value: float) -> Transform:
        if joint.joint_type in _ROTATIONAL:
            return Transform.rotation(value, self._joint_axis(joint, value))
        if joint.joint_type in _TRANSLATIONAL:
            return Transform.translation(0.0, 0.0, value)
        return Transform.identity()

    @staticmethod
    def _joint_axis(joint: Joint, value: float) -> Any:
        from cadgenesis.cad.geometry.core import Vec

        if joint.joint_type == "UNIVERSAL":
            # model the two rotational dof as x then y rotation
            return Vec(1.0, 0.0, 0.0) if abs(value) <= 1e-9 else Vec(0.0, 1.0, 0.0)
        return Vec(0.0, 0.0, 1.0)

    def simulate(
        self,
        mechanism: Mechanism,
        states: dict[str, float],
        t: float = 0.0,
        link_offsets: dict[str, float] | None = None,
    ) -> SimulatedPose:
        """Compute link poses given joint values.

        ``states`` maps joint name -> commanded value.  ``link_offsets`` maps
        a link name to its length along the parent joint's local x-axis so
        serial chains produce visible motion.  The chain is walked in joint
        insertion order; a joint whose link is not yet placed falls back to
        the base frame.
        """
        offsets = link_offsets or {}
        poses: dict[str, Transform] = {}
        for link in mechanism.links:
            poses[link] = self.base
        for joint in mechanism.joints:
            value = states.get(joint.name, 0.0)
            parent = poses.get(joint.link_a, self.base)
            child = parent.composed(self._joint_transform(joint, value))
            length = offsets.get(joint.link_b, 0.0)
            if length:
                child = child.composed(Transform.translation(length, 0.0, 0.0))
            poses[joint.link_b] = child
        return SimulatedPose(time=t, joint_values=dict(states), link_poses=poses)

    def simulate_trajectory(
        self,
        mechanism: Mechanism,
        start: dict[str, float],
        end: dict[str, float],
        steps: int = 10,
        link_offsets: dict[str, float] | None = None,
    ) -> list[SimulatedPose]:
        """Linearly interpolate joint-space and simulate each step."""
        if steps < 1:
            raise ValueError("steps must be >= 1")
        names = list(start)
        poses: list[SimulatedPose] = []
        for step in range(steps + 1):
            alpha = step / steps
            values = {
                name: start[name] + (end.get(name, start[name]) - start[name]) * alpha
                for name in names
            }
            poses.append(self.simulate(mechanism, values, t=alpha, link_offsets=link_offsets))
        return poses

    def check_path(
        self,
        moving: WorldObject,
        path: list[Transform],
        obstacles: list[WorldObject],
        reasoner: SpatialReasoner | None = None,
        margin_mm: float = 0.0,
    ) -> dict[str, Any]:
        """Collision check a moving object along a pose path.

        The object is temporarily repositioned at each sample; its world AABB
        is compared against every obstacle's world AABB.
        """
        reasoner = reasoner or SpatialReasoner()
        collisions: list[dict[str, Any]] = []
        original = moving.pose
        try:
            for i, pose in enumerate(path):
                moving.pose = pose
                for obstacle in obstacles:
                    if obstacle.object_id == moving.object_id:
                        continue
                    min_a, max_a = reasoner.world_bounds(moving)
                    min_b, max_b = reasoner.world_bounds(obstacle)
                    overlap = not (
                        max_a.x <= min_b.x + margin_mm
                        or max_b.x <= min_a.x + margin_mm
                        or max_a.y <= min_b.y + margin_mm
                        or max_b.y <= min_a.y + margin_mm
                        or max_a.z <= min_b.z + margin_mm
                        or max_b.z <= min_a.z + margin_mm
                    )
                    if overlap:
                        collisions.append({"sample": i, "obstacle": obstacle.object_id})
        finally:
            moving.pose = original
        return {
            "collision_free": not collisions,
            "collisions": collisions,
            "samples": len(path),
            "obstacles": len(obstacles),
        }


__all__ = ["JointState", "MotionSimulator", "SimulatedPose"]
