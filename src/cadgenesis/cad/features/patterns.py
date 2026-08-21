"""cadgenesis.cad.features.patterns
===============================
Pattern and mirror features: linear pattern, circular pattern and mirror.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.cad.features.base import (
    Feature,
    FeatureOperation,
    FeatureType,
    register_feature,
)


class _PatternFeature(Feature):
    operation = FeatureOperation.NEUTRAL

    def validate(self) -> list[str]:
        problems = super().validate()
        if not self.references:
            problems.append(f"{self.type.value.lower()} requires a source feature reference")
        return problems


@register_feature
class LinearPattern(_PatternFeature):
    type = FeatureType.LINEAR_PATTERN

    def __init__(
        self,
        name: str,
        sketch_ref: str = "",
        params: dict[str, Any] | None = None,
        references: list[str] | None = None,
        description: str = "",
    ) -> None:
        params = dict(params or {})
        params.setdefault("count_x", 2)
        params.setdefault("spacing_x", 10.0)
        params.setdefault("count_y", 1)
        params.setdefault("spacing_y", 10.0)
        super().__init__(name, sketch_ref, params, references, description)

    def validate(self) -> list[str]:
        problems = super().validate()
        if int(self.params.get("count_x", 1)) < 1 or int(self.params.get("count_y", 1)) < 1:
            problems.append("pattern counts must be >= 1")
        if (
            float(self.params.get("spacing_x", 0)) <= 0
            and float(self.params.get("spacing_y", 0)) <= 0
        ):
            problems.append("at least one pattern spacing must be positive")
        return problems


@register_feature
class CircularPattern(_PatternFeature):
    type = FeatureType.CIRCULAR_PATTERN

    def __init__(
        self,
        name: str,
        sketch_ref: str = "",
        params: dict[str, Any] | None = None,
        references: list[str] | None = None,
        description: str = "",
    ) -> None:
        params = dict(params or {})
        params.setdefault("count", 6)
        params.setdefault("axis", (0.0, 0.0, 1.0))
        super().__init__(name, sketch_ref, params, references, description)

    def validate(self) -> list[str]:
        problems = super().validate()
        if int(self.params.get("count", 1)) < 1:
            problems.append("circular pattern count must be >= 1")
        return problems


@register_feature
class Mirror(_PatternFeature):
    type = FeatureType.MIRROR

    def __init__(
        self,
        name: str,
        sketch_ref: str = "",
        params: dict[str, Any] | None = None,
        references: list[str] | None = None,
        description: str = "",
    ) -> None:
        params = dict(params or {})
        params.setdefault("plane", "XY")
        super().__init__(name, sketch_ref, params, references, description)

    def validate(self) -> list[str]:
        problems = super().validate()
        if self.params.get("plane") not in ("XY", "XZ", "YZ"):
            problems.append(f"unknown mirror plane {self.params.get('plane')!r}")
        return problems


__all__ = ["CircularPattern", "LinearPattern", "Mirror"]
