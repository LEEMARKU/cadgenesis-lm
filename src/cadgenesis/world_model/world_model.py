"""cadgenesis.world_model.world_model
====================================
World-model facade (Pillar 4).

:class:`WorldModelSystem` is the single entry point for the world model.  It
owns the object graph and every reasoner (spatial, mechanical, functional,
assembly, affordances, intent, simulation, planning) and exposes a unified
``reason`` capability API, full snapshot/restore and optional wiring into the
Pillar-3 multimodal system and the memory system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cadgenesis.cad.geometry.core import Transform
from cadgenesis.world_model.affordances import AffordanceMapper
from cadgenesis.world_model.assembly import AssemblyValidator, WorldAssembly
from cadgenesis.world_model.design_intent import (
    DesignIntent,
    DesignIntentCapture,
    IntentAnnotation,
)
from cadgenesis.world_model.functional import FunctionalReasoner
from cadgenesis.world_model.mechanical import MechanicalReasoner
from cadgenesis.world_model.objects import (
    Material,
    ObjectGraph,
    WorldObject,
    make_object,
)
from cadgenesis.world_model.planning import (
    WorldModelPlanner,
)
from cadgenesis.world_model.simulator import MotionSimulator
from cadgenesis.world_model.spatial import SpatialReasoner


@dataclass
class WorldModelState:
    """Full serializable state of the world model."""

    objects: list[WorldObject] = field(default_factory=list)
    assembly: dict[str, Any] = field(default_factory=dict)
    intents: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objects": [o.to_dict() for o in self.objects],
            "assembly": dict(self.assembly),
            "intents": [dict(i) for i in self.intents],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorldModelState:
        return cls(
            objects=[WorldObject.from_dict(o) for o in data.get("objects", [])],
            assembly=dict(data.get("assembly", {})),
            intents=[dict(i) for i in data.get("intents", [])],
        )


class WorldModelSystem:
    """Facade for all world-model reasoning capabilities."""

    def __init__(self, name: str = "world") -> None:
        self.name = name
        self.graph = ObjectGraph()
        self.spatial = SpatialReasoner()
        self.mechanical = MechanicalReasoner()
        self.functional = FunctionalReasoner()
        self.assembly_validator = AssemblyValidator()
        self.affordances = AffordanceMapper()
        self.intent = DesignIntentCapture()
        self.simulator = MotionSimulator()
        self.planner = WorldModelPlanner()
        self.intents: list[DesignIntent] = []
        self.multimodal: Any = None
        self.memory: Any = None

    # --------------------------------------------------------------- graph

    def add_object(
        self,
        feature: str,
        name: str,
        parameters: dict[str, Any] | None = None,
        material: str | Material | None = None,
        **kwargs: Any,
    ) -> WorldObject:
        obj = make_object(feature, name, parameters, material=material, **kwargs)
        self.graph.add(obj)
        return obj

    def add_world_object(self, obj: WorldObject) -> WorldObject:
        self.graph.add(obj)
        return obj

    def relate(self, parent_id: str, child_id: str, relation: str = "mounts") -> None:
        self.graph.relate(parent_id, child_id, relation)

    def pose(self, object_id: str, pose: Transform | dict[str, Any]) -> None:
        self.graph.set_pose(object_id, pose)

    # ------------------------------------------------------------- reason

    def reason(self, capability: str, **kwargs: Any) -> Any:
        """Dispatch a reasoning query by capability name."""
        dispatch = {
            "clearance": lambda: self.spatial.clearance_report(
                kwargs["a"], kwargs["b"], kwargs["minimum"], kwargs.get("axis", "z")
            ),
            "overlap": lambda: self.spatial.overlap(kwargs["a"], kwargs["b"]),
            "fits_inside": lambda: self.spatial.fits_inside(kwargs["outer"], kwargs["inner"]),
            "distance": lambda: self.spatial.distance_between(kwargs["a"], kwargs["b"]),
            "safety": lambda: self.mechanical.check_load(
                kwargs["object"],
                kwargs["load_case"],
                kwargs.get("target_safety_factor"),
            ),
            "stability": lambda: self.mechanical.stability(kwargs["object"]),
            "mass": lambda: self.mechanical.mass_budget(
                kwargs.get("objects", self.graph.objects),
                kwargs.get("limit_kg"),
            ),
            "dof": lambda: self.functional.requires_dof(kwargs["object"], kwargs["minimum"]),
            "affordances": lambda: self.affordances.affordances(kwargs["object"]),
            "supports_action": lambda: self.affordances.supports(
                kwargs["object"], kwargs["action"]
            ),
            "assembly": lambda: self.assembly_validator.validate(
                kwargs.get("assembly", self._as_assembly())
            ),
            "simulate": lambda: self.simulator.simulate(
                kwargs["mechanism"],
                kwargs["states"],
                kwargs.get("t", 0.0),
                kwargs.get("link_offsets"),
            ),
            "check_path": lambda: self.simulator.check_path(
                kwargs["moving"],
                kwargs["path"],
                kwargs.get("obstacles", []),
                kwargs.get("reasoner"),
                kwargs.get("margin_mm", 0.0),
            ),
            "plan": lambda: self.planner.plan(kwargs["goal"]),
            "execute_plan": lambda: self.planner.execute(
                kwargs["plan"], self.graph, kwargs.get("material", "steel")
            ),
        }
        if capability not in dispatch:
            raise KeyError(f"unknown capability {capability!r}; expected one of {sorted(dispatch)}")
        return dispatch[capability]()

    # --------------------------------------------------------------- state

    def _as_assembly(self) -> WorldAssembly:
        return WorldAssembly(name=self.name, parts=self.graph.objects)

    def snapshot(self) -> WorldModelState:
        """Serialize the full world model state."""
        return WorldModelState(
            objects=list(self.graph.objects),
            assembly=self._as_assembly().to_dict(),
            intents=[i.to_dict() for i in self.intents],
        )

    def restore(self, state: WorldModelState | dict[str, Any]) -> None:
        if isinstance(state, dict):
            state = WorldModelState.from_dict(state)
        self.graph = ObjectGraph(objects=list(state.objects))
        self.intents = [
            DesignIntent(
                name=i.get("name", "intent"),
                goals=list(i.get("goals", [])),
                annotations=[
                    IntentAnnotation(
                        target=a.get("target", ""),
                        kind=a.get("kind", "note"),
                        text=a.get("text", ""),
                    )
                    for a in i.get("annotations", [])
                ],
            )
            for i in state.intents
        ]

    # ------------------------------------------------------------ multimodal

    def connect_multimodal(self, multimodal: Any) -> None:
        """Attach the Pillar-3 multimodal system for cross-modal reasoning."""
        self.multimodal = multimodal

    def connect_memory(self, memory: Any) -> None:
        """Attach the memory system for persistence / retrieval."""
        self.memory = memory

    def persist(self) -> str | None:
        """Store the world snapshot in the memory system.

        The world snapshot is design state, so it lands in the ``project``
        pool (the ``world_model`` name is not a registered semantic pool).
        """
        if self.memory is None:
            return None
        key = self.memory.remember("project", self.name, self.snapshot().to_dict())
        return key


__all__ = ["WorldModelState", "WorldModelSystem"]
