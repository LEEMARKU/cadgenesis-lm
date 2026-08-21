"""cadgenesis.cad.features.dress
=============================
Dress-up features: fillet, chamfer and draft.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.cad.features.base import (
    Feature,
    FeatureOperation,
    FeatureType,
    register_feature,
)


@register_feature
class Fillet(Feature):
    type = FeatureType.FILLET
    operation = FeatureOperation.NEUTRAL

    def __init__(
        self,
        name: str,
        sketch_ref: str = "",
        params: dict[str, Any] | None = None,
        references: list[str] | None = None,
        description: str = "",
    ) -> None:
        params = dict(params or {})
        params.setdefault("radius", 1.0)
        params.setdefault("edges", [])
        super().__init__(name, sketch_ref, params, references, description)

    def validate(self) -> list[str]:
        problems = super().validate()
        if float(self.params.get("radius", 0)) <= 0:
            problems.append("fillet radius must be positive")
        if not self.params.get("edges"):
            problems.append("fillet requires at least one edge reference")
        return problems


@register_feature
class Chamfer(Feature):
    type = FeatureType.CHAMFER
    operation = FeatureOperation.NEUTRAL

    def __init__(
        self,
        name: str,
        sketch_ref: str = "",
        params: dict[str, Any] | None = None,
        references: list[str] | None = None,
        description: str = "",
    ) -> None:
        params = dict(params or {})
        params.setdefault("distance", 1.0)
        params.setdefault("edges", [])
        super().__init__(name, sketch_ref, params, references, description)

    def validate(self) -> list[str]:
        problems = super().validate()
        if float(self.params.get("distance", 0)) <= 0:
            problems.append("chamfer distance must be positive")
        if not self.params.get("edges"):
            problems.append("chamfer requires at least one edge reference")
        return problems


@register_feature
class Draft(Feature):
    type = FeatureType.DRAFT
    operation = FeatureOperation.NEUTRAL

    def __init__(
        self,
        name: str,
        sketch_ref: str = "",
        params: dict[str, Any] | None = None,
        references: list[str] | None = None,
        description: str = "",
    ) -> None:
        params = dict(params or {})
        params.setdefault("angle", 1.0)
        super().__init__(name, sketch_ref, params, references, description)

    def validate(self) -> list[str]:
        problems = super().validate()
        if float(self.params.get("angle", 0)) < 0 or float(self.params.get("angle", 0)) >= 90:
            problems.append("draft angle must be in [0, 90) degrees")
        return problems


__all__ = ["Chamfer", "Draft", "Fillet"]
