"""cadgenesis.agents.fleet
========================
Builds the full 18-agent Pillar 5 fleet.

The eight legacy role agents (planner, geometry, constraint, assembly,
manufacturing, optimization, simulation, validation) are combined with the ten
Pillar 5 specialized agents (material, cost, documentation, safety, memory,
retrieval, user, learning, monitoring, debugging) into a single
:class:`~cadgenesis.agents.registry.AgentRegistry`.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.agents.assembly import AssemblyAgent
from cadgenesis.agents.base import Agent
from cadgenesis.agents.constraint import ConstraintAgent
from cadgenesis.agents.cost import CostAgent
from cadgenesis.agents.debugging import DebuggingAgent
from cadgenesis.agents.documentation import DocumentationAgent
from cadgenesis.agents.geometry import GeometryAgent
from cadgenesis.agents.learning import LearningAgent
from cadgenesis.agents.manufacturing import ManufacturingAgent
from cadgenesis.agents.material import MaterialAgent
from cadgenesis.agents.memory import MemoryAgent
from cadgenesis.agents.monitoring import MonitoringAgent
from cadgenesis.agents.optimization import OptimizationAgent
from cadgenesis.agents.planner import PlannerAgent
from cadgenesis.agents.registry import AgentRegistry
from cadgenesis.agents.retrieval import RetrievalAgent
from cadgenesis.agents.safety import SafetyComplianceAgent
from cadgenesis.agents.simulation import SimulationAgent
from cadgenesis.agents.user import UserInteractionAgent
from cadgenesis.agents.validation import ValidationAgent

FLEET_ROLES = (
    "planner",
    "geometry",
    "constraint",
    "assembly",
    "manufacturing",
    "simulation",
    "optimization",
    "validation",
    "material",
    "cost",
    "documentation",
    "safety",
    "memory",
    "retrieval",
    "user",
    "learning",
    "monitoring",
    "debugging",
)


def build_fleet(
    registry: AgentRegistry | None = None,
    memory: Any = None,
    validator: Any = None,
    material_database: Any = None,
    target_cost: float | None = None,
) -> list[Agent]:
    """Instantiate all 18 fleet agents.

    ``memory`` (a :class:`~cadgenesis.memory.MemorySystem`) is injected into the
    memory/retrieval/learning agents; ``validator`` into safety/validation;
    ``material_database`` into material/cost.
    """
    agents: list[Agent] = [
        PlannerAgent(),
        GeometryAgent(),
        ConstraintAgent(),
        AssemblyAgent(),
        ManufacturingAgent(),
        SimulationAgent(),
        OptimizationAgent(target_cost=target_cost),
        ValidationAgent(validator=validator),
        MaterialAgent(database=material_database),
        CostAgent(database=material_database),
        DocumentationAgent(),
        SafetyComplianceAgent(validator=validator),
        MemoryAgent(memory=memory),
        RetrievalAgent(memory=memory),
        UserInteractionAgent(),
        LearningAgent(memory=memory),
        MonitoringAgent(),
        DebuggingAgent(),
    ]
    if registry is not None:
        registry.register_many(agents)
    return agents


def create_fleet_registry(
    memory: Any = None,
    validator: Any = None,
    material_database: Any = None,
) -> AgentRegistry:
    """Build a populated :class:`AgentRegistry` with all 18 agents."""
    registry = AgentRegistry()
    build_fleet(
        registry=registry,
        memory=memory,
        validator=validator,
        material_database=material_database,
    )
    return registry
