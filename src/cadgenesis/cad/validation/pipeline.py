"""cadgenesis.cad.validation.pipeline
===================================
The CAD validation pipeline: runs geometry, topology, GD&T, material and
manufacturability checks over a design object and produces a
:class:`CadValidationReport`.

The pipeline reuses the existing reasoning toolkit (``TopologyAnalyzer``,
``ManufacturingRules``, ``GeometryReasoner``) through :mod:`checks`.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.cad.validation import checks
from cadgenesis.cad.validation.report import CadValidationReport


class CadValidator:
    """Orchestrates CAD validation checks over a ``design`` object."""

    def __init__(
        self,
        analyze_topology: bool = True,
        check_material: bool = True,
        check_gdt: bool = True,
        check_manufacturing: bool = True,
        check_constraints: bool = True,
        check_consistency: bool = True,
    ) -> None:
        self.analyze_topology = analyze_topology
        self.check_material = check_material
        self.check_gdt = check_gdt
        self.check_manufacturing = check_manufacturing
        self.check_constraints = check_constraints
        self.check_consistency = check_consistency
        self._extra_checks: list[Any] = []

    def add_check(self, check_fn: Any) -> None:
        """Register a callable ``design -> list[CadCheckResult]``."""
        self._extra_checks.append(check_fn)

    def validate(self, design: Any) -> CadValidationReport:
        """Validate a CAD design object.

        Supported design shapes (duck-typed):

        - ``design.faces`` / ``design.vertices``  -> mesh topology + quality
        - ``design.validate()``                  -> B-Rep solid check
        - ``design.gdt_spec``                    -> GD&T check
        - ``design.material``                    -> material check
        - ``design.part``                        -> DFM check
        - ``design.check()``                     -> custom model check
        """
        report = CadValidationReport()

        # mesh topology --------------------------------------------------------
        faces = getattr(design, "faces", None)
        vertices = getattr(design, "vertices", None)
        if self.analyze_topology and faces is not None:
            report.results.extend(checks.check_mesh_topology(faces))
            if vertices is not None:
                report.results.extend(checks.check_mesh_quality(vertices, faces))

        # B-Rep solid -----------------------------------------------------------
        validate_fn = getattr(design, "validate", None)
        if callable(validate_fn) and hasattr(design, "face_count") is False:
            # B-Rep solids expose validate(); mesh objects expose validate too so
            # we only treat it as a B-Rep check when faces are not present.
            report.results.extend(checks.check_brep_solid(design))

        # GD&T ------------------------------------------------------------------
        gdt_spec = getattr(design, "gdt_spec", None)
        if self.check_gdt:
            report.results.extend(checks.check_gdt_spec(gdt_spec))

        # material ---------------------------------------------------------------
        material = getattr(design, "material", None)
        if self.check_material and isinstance(material, dict):
            report.results.extend(checks.check_material(material))

        # manufacturability --------------------------------------------------------
        part = getattr(design, "part", None)
        if self.check_manufacturing and isinstance(part, dict):
            report.results.extend(checks.check_manufacturability(part))

        # constraints ----------------------------------------------------------------
        sketch = getattr(design, "sketch", None)
        if self.check_constraints and sketch is not None:
            report.results.extend(checks.check_constraints(sketch))

        # design consistency -----------------------------------------------------------
        if self.check_consistency:
            report.results.extend(checks.check_design_consistency(design))

        # extra custom checks ------------------------------------------------------
        for check_fn in self._extra_checks:
            report.results.extend(check_fn(design))

        return report

    def to_report(self, design: Any) -> CadValidationReport:
        return self.validate(design)


__all__ = ["CadValidator"]
