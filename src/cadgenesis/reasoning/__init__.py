"""cadgenesis.reasoning
====================
Neuro-Symbolic Reasoning Engine for CADGenesis-LM v6.0.

Canonical modules: rule engine (forward + backward chaining, versioning),
constraint solver (solve/propagate/conflict-detect/repair), geometry reasoner,
knowledge graph, manufacturing (DFM) rules, engineering standards
(ISO/ASME/DIN/ANSI/company), workflow + symbolic planners, symbolic reasoner,
topology analyzer (incl. adjacency/connectivity reasoning), the design
validator orchestrator and the hybrid neuro-symbolic reasoning pipeline.
"""

from cadgenesis.reasoning.constraint_solver import (
    Constraint,
    ConstraintSolver,
    Solution,
    Variable,
)
from cadgenesis.reasoning.geometry_reasoner import (
    GeometryReasoner,
    GeometryValidation,
    Primitive,
)
from cadgenesis.reasoning.hybrid import (
    HybridReasoningPipeline,
    HybridReasoningReport,
    StageReport,
)
from cadgenesis.reasoning.knowledge_graph import (
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
)
from cadgenesis.reasoning.manufacturing_rules import (
    ManufacturingAssessment,
    ManufacturingRules,
    MfgCheck,
)
from cadgenesis.reasoning.neuro_symbolic import NeuroSymbolicReasoningEngine
from cadgenesis.reasoning.planner import CADPlan, PlanningStep, TaskPlanner
from cadgenesis.reasoning.rule_engine import (
    Proof,
    Rule,
    RuleEngine,
    RuleResult,
    make_rule,
)
from cadgenesis.reasoning.standards import (
    Standard,
    StandardsCheck,
    StandardsLibrary,
    build_standards_graph,
    default_standards_library,
)
from cadgenesis.reasoning.symbolic_planner import (
    PlanningOperator,
    SymbolicPlan,
    SymbolicPlanner,
)
from cadgenesis.reasoning.symbolic_reasoner import (
    SymbolicExpression,
    SymbolicReasoner,
    VerificationResult,
)
from cadgenesis.reasoning.topology import TopologyAnalyzer, TopologyStats
from cadgenesis.reasoning.validator import (
    CheckResult,
    DesignValidator,
    ValidationReport,
)

__all__ = [
    "CADPlan",
    "CheckResult",
    "Constraint",
    "ConstraintSolver",
    "DesignValidator",
    "GeometryReasoner",
    "GeometryValidation",
    "GraphEdge",
    "GraphNode",
    "HybridReasoningPipeline",
    "HybridReasoningReport",
    "KnowledgeGraph",
    "ManufacturingAssessment",
    "ManufacturingRules",
    "MfgCheck",
    "NeuroSymbolicReasoningEngine",
    "PlanningOperator",
    "PlanningStep",
    "Primitive",
    "Proof",
    "Rule",
    "RuleEngine",
    "RuleResult",
    "Solution",
    "StageReport",
    "Standard",
    "StandardsCheck",
    "StandardsLibrary",
    "SymbolicExpression",
    "SymbolicPlan",
    "SymbolicPlanner",
    "SymbolicReasoner",
    "TaskPlanner",
    "TopologyAnalyzer",
    "TopologyStats",
    "ValidationReport",
    "Variable",
    "VerificationResult",
    "build_standards_graph",
    "default_standards_library",
    "make_rule",
]
