"""
cadgenesis.agents
=================
Multi-Agent Intelligence for CADGenesis-LM v6.0.

Three layers coexist:

* **Embedded** — :class:`MultiAgentSystem` / :class:`InternalAgentRole`
  (torch agent heads inside the transformer, conditioned on memory pools).
* **Orchestration (legacy)** — the pure-Python agent team:
  :class:`AgentCoordinator`, :class:`MessageBus`, :class:`SharedMemory`,
  :class:`TaskScheduler`, :class:`ConsensusEngine` and the eight role agents.
* **Pillar 5 (v6.0)** — the production platform: :class:`AgentBase` lifecycle,
  :class:`AgentRegistry`, :class:`AgentLoader`, plugins, versioning,
  :class:`AgentHealthMonitor`, :class:`EventBus`, DAG scheduling,
  layered shared memory, extended consensus, the task-planning
  :class:`TaskPlanningPipeline`, the :class:`AgentPlatform` facade and an
  18-agent fleet (10 new specialized agents + the 8 legacy role agents).
"""

from cadgenesis.agents.assembly import AssemblyAgent
from cadgenesis.agents.base import Agent, AgentRequest, AgentResult
from cadgenesis.agents.consensus import AgentOpinion, ConsensusEngine
from cadgenesis.agents.constraint import ConstraintAgent
from cadgenesis.agents.coordinator import AgentCoordinator
from cadgenesis.agents.cost import CostAgent
from cadgenesis.agents.debugging import DebuggingAgent
from cadgenesis.agents.design import (
    CostEstimatorAgent,
    DesignIteration,
    DesignOrchestrationLoop,
    DesignReport,
    DesignSwarm,
    DFMManufacturingAgent,
    FEAStressAgent,
    LeadArchitectAgent,
    ReinforcementPolicy,
    build_design_swarm,
)
from cadgenesis.agents.documentation import DocumentationAgent
from cadgenesis.agents.event_bus import Event, EventBus, SharedEventStore
from cadgenesis.agents.fleet import FLEET_ROLES, build_fleet, create_fleet_registry
from cadgenesis.agents.geometry import GeometryAgent
from cadgenesis.agents.health import AgentHealthMonitor, AgentHealthStatus
from cadgenesis.agents.infrastructure import (
    AgentBase,
    AgentLifecycleManager,
    AgentMetadata,
    AgentState,
    Capability,
)
from cadgenesis.agents.integration import ExecutionAdapter
from cadgenesis.agents.learning import LearningAgent
from cadgenesis.agents.loader import AgentLoader, AgentLoadError
from cadgenesis.agents.manufacturing import ManufacturingAgent
from cadgenesis.agents.material import MaterialAgent
from cadgenesis.agents.memory import MemoryAgent
from cadgenesis.agents.message_bus import AgentMessage, MessageBus
from cadgenesis.agents.monitoring import MonitoringAgent
from cadgenesis.agents.multi_agent_system import InternalAgentRole, MultiAgentSystem
from cadgenesis.agents.optimization import OptimizationAgent
from cadgenesis.agents.orchestrator import AgentPlatform
from cadgenesis.agents.pipeline import (
    AgentAssigner,
    IntentAnalyser,
    PipelineReport,
    ResultAggregator,
    TaskDecomposer,
    TaskGraphBuilder,
    TaskPlanningPipeline,
    TaskValidator,
)
from cadgenesis.agents.planner import PlannerAgent
from cadgenesis.agents.plugins import AgentPlugin, PluginManifest
from cadgenesis.agents.registry import AgentRegistry, RegistryError
from cadgenesis.agents.retrieval import RetrievalAgent
from cadgenesis.agents.safety import SafetyComplianceAgent
from cadgenesis.agents.scheduler import AgentTask, TaskScheduler
from cadgenesis.agents.scheduling import (
    DAGScheduler,
    DeadlineScheduler,
    DynamicScheduler,
    LoadBalancer,
    PriorityScheduler,
    SchedulerStats,
    TaskGraph,
    TaskNode,
    WorkerPool,
)
from cadgenesis.agents.shared_memory import LayeredSharedMemory, SharedMemory
from cadgenesis.agents.simulation import SimulationAgent
from cadgenesis.agents.user import UserInteractionAgent
from cadgenesis.agents.validation import ValidationAgent
from cadgenesis.agents.versioning import AgentVersion

__all__ = [
    "FLEET_ROLES",
    "Agent",
    "AgentAssigner",
    "AgentBase",
    "AgentCoordinator",
    "AgentHealthMonitor",
    "AgentHealthStatus",
    "AgentLifecycleManager",
    "AgentLoadError",
    "AgentLoader",
    "AgentMessage",
    "AgentMetadata",
    "AgentOpinion",
    "AgentPlatform",
    "AgentPlugin",
    "AgentRegistry",
    "AgentRequest",
    "AgentResult",
    "AgentState",
    "AgentTask",
    "AgentVersion",
    "AssemblyAgent",
    "Capability",
    "ConsensusEngine",
    "ConstraintAgent",
    "CostAgent",
    "CostEstimatorAgent",
    "DAGScheduler",
    "DFMManufacturingAgent",
    "DeadlineScheduler",
    "DebuggingAgent",
    "DesignIteration",
    "DesignOrchestrationLoop",
    "DesignReport",
    "DesignSwarm",
    "DocumentationAgent",
    "DynamicScheduler",
    "Event",
    "EventBus",
    "ExecutionAdapter",
    "FEAStressAgent",
    "GeometryAgent",
    "IntentAnalyser",
    "InternalAgentRole",
    "LayeredSharedMemory",
    "LeadArchitectAgent",
    "LearningAgent",
    "LoadBalancer",
    "ManufacturingAgent",
    "MaterialAgent",
    "MemoryAgent",
    "MessageBus",
    "MonitoringAgent",
    "MultiAgentSystem",
    "OptimizationAgent",
    "PipelineReport",
    "PlannerAgent",
    "PluginManifest",
    "PriorityScheduler",
    "RegistryError",
    "ReinforcementPolicy",
    "ResultAggregator",
    "RetrievalAgent",
    "SafetyComplianceAgent",
    "SchedulerStats",
    "SharedEventStore",
    "SharedMemory",
    "SimulationAgent",
    "TaskDecomposer",
    "TaskGraph",
    "TaskGraphBuilder",
    "TaskNode",
    "TaskPlanningPipeline",
    "TaskScheduler",
    "TaskValidator",
    "UserInteractionAgent",
    "ValidationAgent",
    "WorkerPool",
    "build_design_swarm",
    "build_fleet",
    "create_fleet_registry",
]
