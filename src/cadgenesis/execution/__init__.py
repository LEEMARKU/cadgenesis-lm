"""
cadgenesis.execution
===================
CAD Execution Intelligence Engine for CADGenesis-LM v2.0.

Pillar 8 execution layer: the engine orchestrates intent → program →
execute → validate → simulate → optimize → repair → export → feedback over
the analytic CAD substrate.
"""

from cadgenesis.execution.cost_estimation import CostBreakdown, CostEstimator
from cadgenesis.execution.execution_engine import CADExecutionEngine, CADExecutionResult
from cadgenesis.execution.exporter import (
    ALL_FORMATS,
    REAL_FORMATS,
    SCRIPT_FORMATS,
    STRUCTURED_FORMATS,
    ExportEngine,
)
from cadgenesis.execution.feedback import FeedbackItem, FeedbackLoop
from cadgenesis.execution.freecad_engine import FreeCADEngine
from cadgenesis.execution.geometry_validation import (
    GeometryCheck,
    GeometryValidationReport,
    GeometryValidator,
)
from cadgenesis.execution.ir_execution import (
    IRExecutionEngine,
    IRExecutionResult,
    IRExecutionState,
    IRObjectState,
    execution_diff,
)
from cadgenesis.execution.manufacturing import (
    ManufacturabilityAnalyzer,
    ManufacturingCheck,
    ManufacturingReport,
)
from cadgenesis.execution.opencascade_engine import OpenCascadeEngine
from cadgenesis.execution.optimization import (
    OBJECTIVES,
    OptimizationEngine,
    OptimizationReport,
)
from cadgenesis.execution.simulation import (
    ANALYSIS_TYPES,
    SimulationEngine,
    SimulationResult,
)
from cadgenesis.execution.topology_analysis import (
    TopologyAnalysisReport,
    TopologyAnalyzer,
    TopologyCheck,
)

__all__ = [
    "ALL_FORMATS",
    "ANALYSIS_TYPES",
    "OBJECTIVES",
    "REAL_FORMATS",
    "SCRIPT_FORMATS",
    "STRUCTURED_FORMATS",
    "CADExecutionEngine",
    "CADExecutionResult",
    "CostBreakdown",
    "CostEstimator",
    "ExportEngine",
    "FeedbackItem",
    "FeedbackLoop",
    "FreeCADEngine",
    "GeometryCheck",
    "GeometryValidationReport",
    "GeometryValidator",
    "IRExecutionEngine",
    "IRExecutionResult",
    "IRExecutionState",
    "IRObjectState",
    "ManufacturabilityAnalyzer",
    "ManufacturingCheck",
    "ManufacturingReport",
    "OpenCascadeEngine",
    "OptimizationEngine",
    "OptimizationReport",
    "SimulationEngine",
    "SimulationResult",
    "TopologyAnalysisReport",
    "TopologyAnalyzer",
    "TopologyCheck",
    "execution_diff",
]
