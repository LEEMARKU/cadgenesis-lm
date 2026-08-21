"""cadgenesis.reasoning.manufacturing_rules
==========================================
Design-for-Manufacturing (DFM) rule checks.

Each check takes a dictionary of part parameters and returns a structured
:class:`MfgCheck` with pass/fail, severity, a human-readable detail and an
actionable recommendation.  Thresholds are industry-typical defaults that can
be overridden per check via the constructor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MfgCheck:
    """Result of a single DFM rule check."""

    check: str
    passed: bool
    severity: str = "error"
    detail: str = ""
    recommendation: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def is_passed(self) -> bool:
        return self.passed


@dataclass
class ManufacturingAssessment:
    """Aggregated DFM result for a part."""

    checks: list[MfgCheck]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def errors(self) -> list[MfgCheck]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warnings(self) -> list[MfgCheck]:
        return [c for c in self.checks if not c.passed and c.severity != "error"]

    def summary(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "total": len(self.checks),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "failed": [c.check for c in self.checks if not c.passed],
        }


class ManufacturingRules:
    """DFM heuristics for common manufacturing processes."""

    def __init__(
        self,
        min_wall_thickness: float = 0.8,
        max_depth_to_diameter: float = 5.0,
        min_hole_diameter: float = 1.0,
        mold_min_wall: float = 0.8,
        mold_max_wall: float = 3.0,
        min_draft_angle: float = 1.0,
        print_min_wall: float = 0.4,
        max_overhang_angle: float = 45.0,
        sheet_min_bend_radius_ratio: float = 1.0,
        cast_min_wall: float = 3.0,
        cast_max_wall: float = 25.0,
        cast_min_draft: float = 1.0,
        weld_min_throat: float = 1.5,
        weld_max_root_gap: float = 2.0,
        tool_min_corner_radius: float = 0.5,
        tool_max_pocket_aspect: float = 6.0,
    ) -> None:
        self.min_wall_thickness = min_wall_thickness
        self.max_depth_to_diameter = max_depth_to_diameter
        self.min_hole_diameter = min_hole_diameter
        self.mold_min_wall = mold_min_wall
        self.mold_max_wall = mold_max_wall
        self.min_draft_angle = min_draft_angle
        self.print_min_wall = print_min_wall
        self.max_overhang_angle = max_overhang_angle
        self.sheet_min_bend_radius_ratio = sheet_min_bend_radius_ratio
        self.cast_min_wall = cast_min_wall
        self.cast_max_wall = cast_max_wall
        self.cast_min_draft = cast_min_draft
        self.weld_min_throat = weld_min_throat
        self.weld_max_root_gap = weld_max_root_gap
        self.tool_min_corner_radius = tool_min_corner_radius
        self.tool_max_pocket_aspect = tool_max_pocket_aspect

    # --------------------------------------------------------------- machining

    def check_machining(self, params: dict[str, Any]) -> list[MfgCheck]:
        checks: list[MfgCheck] = []
        wall = params.get("min_wall_thickness")
        if wall is not None:
            ok = float(wall) >= self.min_wall_thickness
            checks.append(
                MfgCheck(
                    "machining_min_wall",
                    ok,
                    severity="error" if not ok else "info",
                    detail=(f"min wall thickness {wall} mm (limit {self.min_wall_thickness} mm)"),
                    recommendation="Increase wall thickness to avoid chatter/breakage.",
                    params={"value": wall, "limit": self.min_wall_thickness},
                )
            )
        depth = params.get("hole_depth")
        diameter = params.get("hole_diameter")
        if depth is not None and diameter is not None and float(diameter) > 0:
            ratio = float(depth) / float(diameter)
            ok = ratio <= self.max_depth_to_diameter
            checks.append(
                MfgCheck(
                    "machining_depth_ratio",
                    ok,
                    severity="warning" if not ok else "info",
                    detail=(
                        f"hole depth/diameter ratio {ratio:.2f} "
                        f"(limit {self.max_depth_to_diameter})"
                    ),
                    recommendation="Use a stepped or gun drill for deep holes.",
                    params={"ratio": ratio, "limit": self.max_depth_to_diameter},
                )
            )
        if diameter is not None:
            ok = float(diameter) >= self.min_hole_diameter
            checks.append(
                MfgCheck(
                    "machining_min_hole",
                    ok,
                    severity="error" if not ok else "info",
                    detail=f"hole diameter {diameter} mm (limit {self.min_hole_diameter} mm)",
                    recommendation="Standard drill sizes start around 1 mm.",
                    params={"value": diameter, "limit": self.min_hole_diameter},
                )
            )
        if not checks:
            checks.append(
                MfgCheck(
                    "machining_no_params",
                    True,
                    severity="info",
                    detail="No machining parameters supplied; nothing to check.",
                )
            )
        return checks

    # ------------------------------------------------------------- injection

    def check_injection_molding(self, params: dict[str, Any]) -> list[MfgCheck]:
        checks: list[MfgCheck] = []
        wall = params.get("wall_thickness")
        if wall is not None:
            ok = self.mold_min_wall <= float(wall) <= self.mold_max_wall
            checks.append(
                MfgCheck(
                    "mold_wall_thickness",
                    ok,
                    severity="warning" if not ok else "info",
                    detail=(
                        f"wall thickness {wall} mm "
                        f"(target {self.mold_min_wall}-{self.mold_max_wall} mm)"
                    ),
                    recommendation="Keep walls uniform to avoid sink marks and warp.",
                    params={"value": wall},
                )
            )
        draft = params.get("draft_angle")
        if draft is not None:
            ok = float(draft) >= self.min_draft_angle
            checks.append(
                MfgCheck(
                    "mold_draft_angle",
                    ok,
                    severity="warning" if not ok else "info",
                    detail=f"draft angle {draft} deg (min {self.min_draft_angle} deg)",
                    recommendation="Add 1-2 degrees of draft to vertical walls.",
                    params={"value": draft, "limit": self.min_draft_angle},
                )
            )
        if params.get("has_undercut"):
            checks.append(
                MfgCheck(
                    "mold_undercut",
                    False,
                    severity="warning",
                    detail="Part contains an undercut.",
                    recommendation="Add a side-action core or redesign the feature.",
                    params={"value": True},
                )
            )
        if not checks:
            checks.append(
                MfgCheck(
                    "mold_no_params",
                    True,
                    severity="info",
                    detail="No molding parameters supplied; nothing to check.",
                )
            )
        return checks

    # --------------------------------------------------------------- printing

    def check_3d_printing(self, params: dict[str, Any]) -> list[MfgCheck]:
        checks: list[MfgCheck] = []
        wall = params.get("min_wall_thickness")
        if wall is not None:
            ok = float(wall) >= self.print_min_wall
            checks.append(
                MfgCheck(
                    "print_min_wall",
                    ok,
                    severity="warning" if not ok else "info",
                    detail=f"min wall {wall} mm (printable ≥ {self.print_min_wall} mm)",
                    recommendation="Increase wall to avoid thin, fragile shells.",
                    params={"value": wall, "limit": self.print_min_wall},
                )
            )
        overhang = params.get("max_overhang_angle")
        if overhang is not None:
            ok = float(overhang) <= self.max_overhang_angle
            checks.append(
                MfgCheck(
                    "print_overhang",
                    ok,
                    severity="warning" if not ok else "info",
                    detail=f"max overhang {overhang} deg (limit {self.max_overhang_angle} deg)",
                    recommendation="Reduce overhang or add support material.",
                    params={"value": overhang, "limit": self.max_overhang_angle},
                )
            )
        if not checks:
            checks.append(
                MfgCheck(
                    "print_no_params",
                    True,
                    severity="info",
                    detail="No printing parameters supplied; nothing to check.",
                )
            )
        return checks

    # ------------------------------------------------------------- sheet metal

    def check_sheet_metal(self, params: dict[str, Any]) -> list[MfgCheck]:
        checks: list[MfgCheck] = []
        radius = params.get("bend_radius")
        thickness = params.get("material_thickness")
        if radius is not None and thickness is not None and float(thickness) > 0:
            ratio = float(radius) / float(thickness)
            ok = ratio >= self.sheet_min_bend_radius_ratio
            checks.append(
                MfgCheck(
                    "sheet_bend_radius",
                    ok,
                    severity="warning" if not ok else "info",
                    detail=(
                        f"bend radius/thickness ratio {ratio:.2f} "
                        f"(min {self.sheet_min_bend_radius_ratio})"
                    ),
                    recommendation="Use a bend radius at least equal to material thickness.",
                    params={"ratio": ratio, "limit": self.sheet_min_bend_radius_ratio},
                )
            )
        if not checks:
            checks.append(
                MfgCheck(
                    "sheet_no_params",
                    True,
                    severity="info",
                    detail="No sheet-metal parameters supplied; nothing to check.",
                )
            )
        return checks

    # ----------------------------------------------------------------- casting

    def check_casting(self, params: dict[str, Any]) -> list[MfgCheck]:
        checks: list[MfgCheck] = []
        wall = params.get("wall_thickness")
        if wall is not None:
            ok = self.cast_min_wall <= float(wall) <= self.cast_max_wall
            checks.append(
                MfgCheck(
                    "cast_wall_thickness",
                    ok,
                    severity="warning" if not ok else "info",
                    detail=(
                        f"wall thickness {wall} mm "
                        f"(castable {self.cast_min_wall}-{self.cast_max_wall} mm)"
                    ),
                    recommendation=(
                        "Keep walls within the castable range to avoid "
                        "mistun or shrinkage porosity."
                    ),
                    params={"value": wall},
                )
            )
        draft = params.get("draft_angle")
        if draft is not None:
            ok = float(draft) >= self.cast_min_draft
            checks.append(
                MfgCheck(
                    "cast_draft_angle",
                    ok,
                    severity="warning" if not ok else "info",
                    detail=f"draft angle {draft} deg (min {self.cast_min_draft} deg)",
                    recommendation="Add draft so the casting releases from the mold.",
                    params={"value": draft, "limit": self.cast_min_draft},
                )
            )
        if params.get("has_isolated_thick_section"):
            checks.append(
                MfgCheck(
                    "cast_isolated_thick",
                    False,
                    severity="warning",
                    detail="Isolated thick section detected.",
                    recommendation="Add risers or core out the section to avoid shrinkage defects.",
                )
            )
        if params.get("sharp_corners"):
            checks.append(
                MfgCheck(
                    "cast_sharp_corners",
                    False,
                    severity="warning",
                    detail="Sharp corners present in the casting.",
                    recommendation="Add fillets to reduce stress concentration and mold erosion.",
                )
            )
        if not checks:
            checks.append(
                MfgCheck(
                    "cast_no_params",
                    True,
                    severity="info",
                    detail="No casting parameters supplied; nothing to check.",
                )
            )
        return checks

    # ---------------------------------------------------------------- welding

    def check_welding(self, params: dict[str, Any]) -> list[MfgCheck]:
        checks: list[MfgCheck] = []
        throat = params.get("weld_throat")
        if throat is not None:
            ok = float(throat) >= self.weld_min_throat
            checks.append(
                MfgCheck(
                    "weld_throat_size",
                    ok,
                    severity="error" if not ok else "info",
                    detail=f"weld throat {throat} mm (min {self.weld_min_throat} mm)",
                    recommendation="Increase weld size to meet the design load.",
                    params={"value": throat, "limit": self.weld_min_throat},
                )
            )
        gap = params.get("root_gap")
        if gap is not None:
            ok = float(gap) <= self.weld_max_root_gap
            checks.append(
                MfgCheck(
                    "weld_root_gap",
                    ok,
                    severity="warning" if not ok else "info",
                    detail=f"root gap {gap} mm (max {self.weld_max_root_gap} mm)",
                    recommendation="Close the root gap or use a backing bar.",
                    params={"value": gap, "limit": self.weld_max_root_gap},
                )
            )
        if params.get("dissimilar_metals") and params.get("weld_preheat") is None:
            checks.append(
                MfgCheck(
                    "weld_dissimilar",
                    True,
                    severity="warning",
                    detail="Dissimilar metals detected without preheat specification.",
                    recommendation="Specify preheat/interlayer temperature for dissimilar welds.",
                )
            )
        if not checks:
            checks.append(
                MfgCheck(
                    "weld_no_params",
                    True,
                    severity="info",
                    detail="No welding parameters supplied; nothing to check.",
                )
            )
        return checks

    # ----------------------------------------------------------------- tooling

    def check_tooling(self, params: dict[str, Any]) -> list[MfgCheck]:
        checks: list[MfgCheck] = []
        radius = params.get("min_corner_radius")
        if radius is not None:
            ok = float(radius) >= self.tool_min_corner_radius
            checks.append(
                MfgCheck(
                    "tool_min_corner_radius",
                    ok,
                    severity="warning" if not ok else "info",
                    detail=(
                        f"min corner radius {radius} mm (limit {self.tool_min_corner_radius} mm)"
                    ),
                    recommendation="Increase corner radii to use standard cutter sizes.",
                    params={"value": radius, "limit": self.tool_min_corner_radius},
                )
            )
        depth = params.get("pocket_depth")
        width = params.get("pocket_width")
        if depth is not None and width is not None and float(width) > 0:
            aspect = float(depth) / float(width)
            ok = aspect <= self.tool_max_pocket_aspect
            checks.append(
                MfgCheck(
                    "tool_pocket_aspect",
                    ok,
                    severity="warning" if not ok else "info",
                    detail=(
                        f"pocket depth/width aspect {aspect:.2f} "
                        f"(limit {self.tool_max_pocket_aspect})"
                    ),
                    recommendation="Break deep pockets into steps to limit tool deflection.",
                    params={"aspect": aspect, "limit": self.tool_max_pocket_aspect},
                )
            )
        if params.get("undercut_requires_special_tool"):
            checks.append(
                MfgCheck(
                    "tool_undercut",
                    True,
                    severity="warning",
                    detail="Feature requires a special undercut tool.",
                    recommendation="Redesign the undercut or budget for a form tool.",
                )
            )
        if not checks:
            checks.append(
                MfgCheck(
                    "tool_no_params",
                    True,
                    severity="info",
                    detail="No tooling parameters supplied; nothing to check.",
                )
            )
        return checks

    # --------------------------------------------------------------- tolerance

    def check_tolerance(self, params: dict[str, Any]) -> list[MfgCheck]:
        """GD&T / dimensional tolerance feasibility checks."""
        checks: list[MfgCheck] = []
        nominal = params.get("nominal_size")
        tolerance = params.get("tolerance_um")
        if nominal is not None and tolerance is not None and float(nominal) > 0:
            ratio = float(tolerance) / (float(nominal) * 1000.0)
            ok = ratio >= 1e-5
            checks.append(
                MfgCheck(
                    "tolerance_feasibility",
                    ok,
                    severity="warning" if not ok else "info",
                    detail=(f"tolerance {tolerance} µm on {nominal} mm ({ratio:.2e} of nominal)"),
                    recommendation="Loosen the tolerance or switch to grinding/lapping.",
                    params={"value": tolerance, "ratio": ratio},
                )
            )
        if params.get("gd_t_datum") is None and params.get("tolerance_um") is not None:
            checks.append(
                MfgCheck(
                    "tolerance_datum",
                    True,
                    severity="warning",
                    detail="Tolerance specified without an explicit datum reference.",
                    recommendation="Reference a datum on the drawing for repeatable inspection.",
                )
            )
        if params.get("profile_tolerance") and params.get("position_tolerance") is None:
            checks.append(
                MfgCheck(
                    "tolerance_profile_position",
                    True,
                    severity="warning",
                    detail="Profile tolerance without a position/location tolerance.",
                    recommendation="Add a position tolerance to control feature location.",
                )
            )
        if not checks:
            checks.append(
                MfgCheck(
                    "tolerance_no_params",
                    True,
                    severity="info",
                    detail="No tolerance parameters supplied; nothing to check.",
                )
            )
        return checks

    # ------------------------------------------------------------------ assess

    def assess(self, part: dict[str, Any]) -> ManufacturingAssessment:
        """Run every relevant DFM check for a part dictionary.

        ``part`` may carry a ``processes`` list (e.g. ``["machining"]``) to
        select checks; by default all processes are checked.
        """
        processes: list[str] = part.get("processes") or [
            "machining",
            "injection_molding",
            "3d_printing",
            "sheet_metal",
        ]
        check_map: dict[str, Any] = {
            "machining": self.check_machining,
            "injection_molding": self.check_injection_molding,
            "3d_printing": self.check_3d_printing,
            "sheet_metal": self.check_sheet_metal,
            "casting": self.check_casting,
            "welding": self.check_welding,
            "tooling": self.check_tooling,
            "tolerance": self.check_tolerance,
        }
        checks: list[MfgCheck] = []
        for process in processes:
            handler = check_map.get(process)
            if handler is None:
                raise ValueError(f"unknown process {process!r}")
            checks.extend(handler(part))
        return ManufacturingAssessment(checks)


__all__ = [
    "ManufacturingAssessment",
    "ManufacturingRules",
    "MfgCheck",
]
