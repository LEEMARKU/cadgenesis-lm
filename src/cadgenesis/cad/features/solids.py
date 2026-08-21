"""cadgenesis.cad.features.solids
==============================
Sketch-based solid features: extrude, revolve, loft, sweep, rib, thicken,
cut, pocket, hole and shell.

Each feature carries validated parameters and a reference to a sketch
profile.  These are semantic feature descriptions — a CAD kernel backend
(FreeCAD / OpenCASCADE) turns them into actual geometry.
"""

from __future__ import annotations

from cadgenesis.cad.features.base import (
    Feature,
    FeatureOperation,
    FeatureType,
    register_feature,
)


class _SketchFeature(Feature):
    """Common validation for features built from a sketch profile."""

    def validate(self) -> list[str]:
        problems = super().validate()
        if not self.sketch_ref:
            problems.append(f"feature {self.name!r} requires a sketch profile")
        return problems


@register_feature
class Extrude(_SketchFeature):
    type = FeatureType.EXTRUDE
    operation = FeatureOperation.ADDITIVE

    def validate(self) -> list[str]:
        problems = super().validate()
        depth = self.params.get("depth", self.params.get("distance"))
        if depth is None or float(depth) <= 0:
            problems.append("extrude depth must be positive")
        if self.params.get("closed_profile", True) and not self.params.get(
            "profile_is_closed", True
        ):
            problems.append("extrude to a solid requires a closed profile")
        return problems


@register_feature
class Cut(_SketchFeature):
    type = FeatureType.CUT
    operation = FeatureOperation.SUBTRACTIVE

    def validate(self) -> list[str]:
        problems = super().validate()
        depth = self.params.get("depth", self.params.get("distance"))
        if depth is None and not self.params.get("through_all", False):
            problems.append("cut needs a depth or through_all=True")
        if depth is not None and float(depth) <= 0:
            problems.append("cut depth must be positive")
        return problems


@register_feature
class Pocket(_SketchFeature):
    type = FeatureType.POCKET
    operation = FeatureOperation.SUBTRACTIVE

    def validate(self) -> list[str]:
        problems = super().validate()
        if not self.params.get("blind", True) and not self.params.get("through_all", False):
            problems.append("pocket must be blind or through_all")
        return problems


@register_feature
class Revolve(_SketchFeature):
    type = FeatureType.REVOLVE
    operation = FeatureOperation.ADDITIVE

    def validate(self) -> list[str]:
        problems = super().validate()
        angle = self.params.get("angle", 2 * 3.141592653589793)
        if float(angle) <= 0 or float(angle) > 2 * 3.141592653589793:
            problems.append("revolve angle must be in (0, 2*pi]")
        if not self.params.get("axis"):
            problems.append("revolve requires an axis")
        return problems


@register_feature
class Loft(_SketchFeature):
    type = FeatureType.LOFT
    operation = FeatureOperation.ADDITIVE

    def validate(self) -> list[str]:
        problems = super().validate()
        profiles = self.params.get("profiles")
        if not isinstance(profiles, (list, tuple)) or len(profiles) < 2:
            problems.append("loft requires at least 2 profiles")
        return problems


@register_feature
class Sweep(_SketchFeature):
    type = FeatureType.SWEEP
    operation = FeatureOperation.ADDITIVE

    def validate(self) -> list[str]:
        problems = super().validate()
        if not self.params.get("path"):
            problems.append("sweep requires a path")
        return problems


@register_feature
class Rib(_SketchFeature):
    type = FeatureType.RIB
    operation = FeatureOperation.ADDITIVE

    def validate(self) -> list[str]:
        problems = super().validate()
        if self.params.get("thickness") is None or float(self.params.get("thickness", 0)) <= 0:
            problems.append("rib requires a positive thickness")
        return problems


@register_feature
class Thicken(_SketchFeature):
    type = FeatureType.THICKEN
    operation = FeatureOperation.ADDITIVE

    def validate(self) -> list[str]:
        problems = super().validate()
        if float(self.params.get("thickness", 0)) <= 0:
            problems.append("thicken requires a positive thickness")
        return problems


@register_feature
class Hole(_SketchFeature):
    type = FeatureType.HOLE
    operation = FeatureOperation.SUBTRACTIVE

    def validate(self) -> list[str]:
        problems = super().validate()
        if float(self.params.get("diameter", 0)) <= 0:
            problems.append("hole diameter must be positive")
        hole_type = self.params.get("hole_type", "simple")
        if hole_type not in ("simple", "counterbore", "countersink", "tapped"):
            problems.append(f"unknown hole type {hole_type!r}")
        return problems


@register_feature
class Shell(_SketchFeature):
    type = FeatureType.SHELL
    operation = FeatureOperation.NEUTRAL

    def validate(self) -> list[str]:
        problems = super().validate()
        if float(self.params.get("thickness", 0)) <= 0:
            problems.append("shell thickness must be positive")
        return problems


__all__ = [
    "Cut",
    "Extrude",
    "Hole",
    "Loft",
    "Pocket",
    "Revolve",
    "Rib",
    "Shell",
    "Sweep",
    "Thicken",
]
