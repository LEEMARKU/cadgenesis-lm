"""cadgenesis.cad.validation
===========================
CAD validation pipeline: geometry, topology, GD&T, material and
manufacturability checks over design objects, plus report types.
"""

from cadgenesis.cad.validation.checks import (
    check_brep_solid,
    check_constraints,
    check_design_consistency,
    check_gdt_spec,
    check_manufacturability,
    check_material,
    check_mesh_quality,
    check_mesh_topology,
)
from cadgenesis.cad.validation.pipeline import CadValidator
from cadgenesis.cad.validation.report import CadCheckResult, CadValidationReport

__all__ = [
    "CadCheckResult",
    "CadValidationReport",
    "CadValidator",
    "check_brep_solid",
    "check_constraints",
    "check_design_consistency",
    "check_gdt_spec",
    "check_manufacturability",
    "check_material",
    "check_mesh_quality",
    "check_mesh_topology",
]
