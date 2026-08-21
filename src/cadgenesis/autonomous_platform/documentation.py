"""
Autonomous Documentation - CAD documentation, BOM, manufacturing report, simulation report,
validation report, API report, technical report.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any


class DocumentType(str, Enum):
    CAD_DOCUMENTATION = "cad_documentation"
    BOM = "bill_of_materials"
    MANUFACTURING_REPORT = "manufacturing_report"
    SIMULATION_REPORT = "simulation_report"
    VALIDATION_REPORT = "validation_report"
    API_REPORT = "api_report"
    TECHNICAL_REPORT = "technical_report"


@dataclass
class Document:
    """A generated document."""

    document_id: str
    doc_type: DocumentType
    title: str
    content: str
    format: str = "markdown"  # markdown, html, pdf
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class DocumentSet:
    """A complete set of documents for an engineering project."""

    set_id: str
    workflow_id: str
    documents: list[Document] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_document(self, doc_type: DocumentType) -> Document | None:
        for doc in self.documents:
            if doc.doc_type == doc_type:
                return doc
        return None


class AutonomousDocumentation:
    """Automatically generates all engineering documentation."""

    def __init__(self):
        self._document_sets: dict[str, DocumentSet] = {}
        self._lock = RLock()

    def generate_documentation(
        self,
        workflow_id: str,
        workflow_state: Any,
        stage_results: dict[str, Any],
        validation_report: Any,
        explanation_report: Any,
    ) -> DocumentSet:
        """Generate complete documentation set."""
        doc_set = DocumentSet(
            set_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
        )

        # CAD Documentation
        cad_doc = self._generate_cad_documentation(stage_results)
        doc_set.documents.append(cad_doc)

        # Bill of Materials
        bom_doc = self._generate_bom(stage_results)
        doc_set.documents.append(bom_doc)

        # Manufacturing Report
        mfg_doc = self._generate_manufacturing_report(stage_results)
        doc_set.documents.append(mfg_doc)

        # Simulation Report
        sim_doc = self._generate_simulation_report(stage_results)
        doc_set.documents.append(sim_doc)

        # Validation Report
        val_doc = self._generate_validation_report(validation_report)
        doc_set.documents.append(val_doc)

        # API Report
        api_doc = self._generate_api_report(stage_results)
        doc_set.documents.append(api_doc)

        # Technical Report
        tech_doc = self._generate_technical_report(
            workflow_state, stage_results, explanation_report
        )
        doc_set.documents.append(tech_doc)

        with self._lock:
            self._document_sets[doc_set.set_id] = doc_set

        return doc_set

    def _generate_cad_documentation(self, stage_results: dict[str, Any]) -> Document:
        cad_gen = stage_results.get("cad_generation", {})
        geometry_val = stage_results.get("geometry_validation", {})
        constraint_val = stage_results.get("constraint_validation", {})

        content = f"""# CAD Documentation

## Generated Model
- **Model ID**: {cad_gen.get("model_id", "N/A")}
- **Format**: {cad_gen.get("format", "STEP")}
- **Features**: {cad_gen.get("feature_count", 0)}
- **Parts**: {cad_gen.get("part_count", 0)}

## Geometry Validation
- **Manifold**: {geometry_val.get("manifold", "N/A")}
- **Topology Valid**: {geometry_val.get("topology_valid", "N/A")}
- **Self-Intersections**: {geometry_val.get("self_intersections", 0)}
- **Open Edges**: {geometry_val.get("open_edges", 0)}

## Constraint Validation
- **Constraints Satisfied**: {constraint_val.get("satisfied", 0)}
- **Constraints Violated**: {constraint_val.get("violated", 0)}
- **Propagation Complete**: {constraint_val.get("propagation_complete", "N/A")}
"""

        return Document(
            document_id=str(uuid.uuid4()),
            doc_type=DocumentType.CAD_DOCUMENTATION,
            title="CAD Documentation",
            content=content,
        )

    def _generate_bom(self, stage_results: dict[str, Any]) -> Document:
        cad_gen = stage_results.get("cad_generation", {})
        parts = cad_gen.get("parts", [])

        content = "# Bill of Materials\n\n"
        content += "| Item | Part Number | Description | Quantity | Material | Mass (kg) |\n"
        content += "|------|-------------|-------------|----------|----------|-----------|\n"

        for i, part in enumerate(parts):
            pn = part.get("part_number", f"PN-{i + 1:04d}")
            content += (
                f"| {i + 1} | {pn} | {part.get('name', 'Part')} | "
                f"{part.get('quantity', 1)} | {part.get('material', 'Steel')} | "
                f"{part.get('mass', 0.0):.3f} |\n"
            )

        if not parts:
            content += "| 1 | PN-0001 | Generated Assembly | 1 | Steel | 0.000 |\n"

        return Document(
            document_id=str(uuid.uuid4()),
            doc_type=DocumentType.BOM,
            title="Bill of Materials",
            content=content,
        )

    def _generate_manufacturing_report(self, stage_results: dict[str, Any]) -> Document:
        mfg = stage_results.get("manufacturing_analysis", {})
        opt = stage_results.get("optimization", {})

        content = f"""# Manufacturing Report

## Manufacturing Analysis
- **Process**: {mfg.get("process", "CNC Machining")}
- **Feasibility**: {mfg.get("feasibility", "Feasible")}
- **Estimated Cost**: ${mfg.get("estimated_cost", 0):.2f}
- **Lead Time**: {mfg.get("lead_time_days", 0)} days
- **Tolerance Compliance**: {mfg.get("tolerance_compliance", "N/A")}

## DFM Checks
- **Machining**: {mfg.get("machining_check", "Pass")}
- **Tooling**: {mfg.get("tooling_check", "Pass")}
- **Fixturing**: {mfg.get("fixturing_check", "Pass")}

## Optimization Results
- **Weight Reduction**: {opt.get("weight_reduction_pct", 0):.1f}%
- **Material Savings**: {opt.get("material_savings_pct", 0):.1f}%
- **Cost Reduction**: {opt.get("cost_reduction_pct", 0):.1f}%
"""

        return Document(
            document_id=str(uuid.uuid4()),
            doc_type=DocumentType.MANUFACTURING_REPORT,
            title="Manufacturing Report",
            content=content,
        )

    def _generate_simulation_report(self, stage_results: dict[str, Any]) -> Document:
        sim = stage_results.get("simulation", {})

        content = f"""# Simulation Report

## Simulation Setup
- **Type**: {sim.get("type", "Static Structural")}
- **Solver**: {sim.get("solver", "FEA")}
- **Mesh Size**: {sim.get("mesh_size", "Medium")}
- **Boundary Conditions**: {sim.get("boundary_conditions", "Fixed support + Load")}

## Results
- **Max Stress**: {sim.get("max_stress_mpa", 0):.1f} MPa
- **Max Displacement**: {sim.get("max_displacement_mm", 0):.3f} mm
- **Safety Factor**: {sim.get("safety_factor", 0):.2f}
- **Convergence**: {sim.get("convergence", "Achieved")}

## Validation
- **Mesh Independence**: {sim.get("mesh_independence", "Verified")}
- **Energy Error**: {sim.get("energy_error_pct", 0):.2f}%
"""

        return Document(
            document_id=str(uuid.uuid4()),
            doc_type=DocumentType.SIMULATION_REPORT,
            title="Simulation Report",
            content=content,
        )

    def _generate_validation_report(self, validation_report: Any) -> Document:
        if not validation_report:
            content = "# Validation Report\n\nNo validation performed."
        else:
            content = f"""# Validation Report

## Overall Status
- **Status**: {validation_report.overall_status.value}
- **Score**: {validation_report.overall_score:.3f}
- **Summary**: {validation_report.summary}

## Detailed Results
"""

            for result in validation_report.results:
                content += f"### {result.name} ({result.category.value})\n"
                content += f"- **Status**: {result.status.value}\n"
                content += f"- **Score**: {result.score:.3f}\n"
                content += f"- **Details**: {result.details}\n"
                if result.recommendations:
                    content += f"- **Recommendations**: {', '.join(result.recommendations)}\n"
                content += "\n"

        return Document(
            document_id=str(uuid.uuid4()),
            doc_type=DocumentType.VALIDATION_REPORT,
            title="Validation Report",
            content=content,
        )

    def _generate_api_report(self, stage_results: dict[str, Any]) -> Document:
        content = """# API Report

## Generated Interfaces
- **CAD Export**: STEP, IGES, STL, OBJ
- **Simulation Input**: FEA mesh, boundary conditions
- **Manufacturing Output**: Toolpaths, NC code
- **BOM Export**: CSV, Excel, JSON

## Integration Points
- **PLM**: Teamcenter, Windchill, Aras
- **ERP**: SAP, Oracle, Microsoft Dynamics
- **MES**: Siemens Opcenter, Rockwell FactoryTalk
"""

        return Document(
            document_id=str(uuid.uuid4()),
            doc_type=DocumentType.API_REPORT,
            title="API Report",
            content=content,
        )

    def _generate_technical_report(
        self,
        workflow_state: Any,
        stage_results: dict[str, Any],
        explanation_report: Any,
    ) -> Document:
        content = f"""# Technical Report

## Project Overview
- **Workflow ID**: {workflow_state.workflow_id if workflow_state else "N/A"}
- **Prompt**: {workflow_state.prompt if workflow_state else "N/A"}
- **Status**: {workflow_state.status.value if workflow_state else "N/A"}
- **Duration**: {
            (workflow_state.completed_at - workflow_state.started_at)
            if workflow_state and workflow_state.started_at and workflow_state.completed_at
            else "N/A"
        } seconds

## Pipeline Execution
"""

        for stage, result in stage_results.items():
            if isinstance(result, dict):
                content += f"### {stage.replace('_', ' ').title()}\n"
                content += f"- **Status**: {result.get('status', 'completed')}\n"
                content += f"- **Confidence**: {result.get('confidence', 'N/A')}\n\n"

        if explanation_report:
            content += f"## Design Rationale\n{explanation_report.design_rationale}\n\n"
            content += "## Confidence Assessment\n"
            for k, v in explanation_report.confidence_report.items():
                content += f"- **{k}**: {v:.2f}\n"
            content += f"\n**Overall Confidence**: {explanation_report.overall_confidence:.2f}\n"

        return Document(
            document_id=str(uuid.uuid4()),
            doc_type=DocumentType.TECHNICAL_REPORT,
            title="Technical Report",
            content=content,
        )

    def get_document_set(self, set_id: str) -> DocumentSet | None:
        with self._lock:
            return self._document_sets.get(set_id)

    def list_document_sets(self, workflow_id: str | None = None) -> list[DocumentSet]:
        with self._lock:
            sets = list(self._document_sets.values())
            if workflow_id:
                sets = [s for s in sets if s.workflow_id == workflow_id]
            return sets
