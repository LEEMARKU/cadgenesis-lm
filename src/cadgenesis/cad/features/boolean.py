"""cadgenesis.cad.features.boolean
================================
Boolean features: union, subtract and intersect between solid bodies.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.cad.features.base import (
    Feature,
    FeatureOperation,
    FeatureType,
    register_feature,
)


class _BooleanFeature(Feature):
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
        params.setdefault("bodies", [])
        super().__init__(name, sketch_ref, params, references, description)

    def validate(self) -> list[str]:
        problems = super().validate()
        bodies = self.params.get("bodies", [])
        if not isinstance(bodies, (list, tuple)) or len(bodies) < 2:
            problems.append(f"{self.type.value.lower()} requires at least 2 bodies")
        return problems


@register_feature
class BooleanUnion(_BooleanFeature):
    type = FeatureType.BOOLEAN_UNION


@register_feature
class BooleanSubtract(_BooleanFeature):
    type = FeatureType.BOOLEAN_SUBTRACT


@register_feature
class BooleanIntersect(_BooleanFeature):
    type = FeatureType.BOOLEAN_INTERSECT


__all__ = ["BooleanIntersect", "BooleanSubtract", "BooleanUnion"]
