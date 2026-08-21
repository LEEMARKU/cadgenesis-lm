"""cadgenesis.world_model
========================
World-model package (Pillar 4).

A unified, reasoning-friendly representation of a CAD design plus the
reasoners that operate on it:

* :mod:`objects` — the object graph, materials, load cases.
* :mod:`spatial` — clearance, overlap, fit, pose math.
* :mod:`mechanical` — first-order stress / safety / stability / mass.
* :mod:`functional` — DOF, envelope, load paths, flow continuity.
* :mod:`assembly` — mate / joint validation and mobility.
* :mod:`affordances` — interaction affordance mapping.
* :mod:`design_intent` — goals, requirements and rationale capture.
* :mod:`simulator` — forward-kinematics motion and path checks.
* :mod:`planning` — goal plans and their execution against the graph.
* :mod:`world_model` — the :class:`WorldModelSystem` facade.
* :mod:`integration` — bridges to Pillar-3 multimodal, memory and datasets.
"""

from cadgenesis.world_model.affordances import (
    AFFORDANCE_ACTIONS,
    Affordance,
    AffordanceMapper,
)
from cadgenesis.world_model.assembly import (
    AssemblyCheck,
    AssemblyValidator,
    WorldAssembly,
)
from cadgenesis.world_model.design_intent import (
    DesignIntent,
    DesignIntentCapture,
    IntentAnnotation,
)
from cadgenesis.world_model.functional import FunctionalCheck, FunctionalReasoner
from cadgenesis.world_model.integration import WorldModelIntegration
from cadgenesis.world_model.mechanical import MechanicalReasoner, MechanicalResult
from cadgenesis.world_model.objects import (
    PRIMITIVE_FAMILIES,
    STOCK_MATERIALS,
    BoundaryCondition,
    LoadCase,
    Material,
    ObjectGraph,
    WorldObject,
    make_object,
)
from cadgenesis.world_model.planning import (
    ExecutionResult,
    StepOutcome,
    WorldModelPlanner,
    WorldStep,
)
from cadgenesis.world_model.simulator import (
    JointState,
    MotionSimulator,
    SimulatedPose,
)
from cadgenesis.world_model.spatial import SpatialReasoner, SpatialReport
from cadgenesis.world_model.world_model import WorldModelState, WorldModelSystem

__all__ = [
    "AFFORDANCE_ACTIONS",
    "PRIMITIVE_FAMILIES",
    "STOCK_MATERIALS",
    "Affordance",
    "AffordanceMapper",
    "AssemblyCheck",
    "AssemblyValidator",
    "BoundaryCondition",
    "DesignIntent",
    "DesignIntentCapture",
    "ExecutionResult",
    "FunctionalCheck",
    "FunctionalReasoner",
    "IntentAnnotation",
    "JointState",
    "LoadCase",
    "Material",
    "MechanicalReasoner",
    "MechanicalResult",
    "MotionSimulator",
    "ObjectGraph",
    "SimulatedPose",
    "SpatialReasoner",
    "SpatialReport",
    "StepOutcome",
    "WorldAssembly",
    "WorldModelIntegration",
    "WorldModelPlanner",
    "WorldModelState",
    "WorldModelSystem",
    "WorldObject",
    "WorldStep",
    "make_object",
]
