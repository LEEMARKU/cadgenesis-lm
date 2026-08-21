"""cadgenesis.cad.features.base
============================
Feature-based modelling core: the :class:`Feature` base class, feature
registry, ordered :class:`FeatureTree` and :class:`DesignHistory`.

A feature is a named parametric operation applied to a part.  Features
reference sketch profiles and parameters, form parent/child dependencies in
the feature tree, and record every change in the design history so the model
can be replayed (the core of *parametric* CAD understanding).
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


class FeatureType(enum.Enum):
    """The full parametric feature-operation vocabulary."""

    # sketch-based additive
    EXTRUDE = "EXTRUDE"
    REVOLVE = "REVOLVE"
    LOFT = "LOFT"
    SWEEP = "SWEEP"
    RIB = "RIB"
    THICKEN = "THICKEN"
    # sketch-based subtractive
    CUT = "CUT"
    POCKET = "POCKET"
    HOLE = "HOLE"
    SHELL = "SHELL"
    # dress-up
    FILLET = "FILLET"
    CHAMFER = "CHAMFER"
    DRAFT = "DRAFT"
    # pattern / transform
    MIRROR = "MIRROR"
    LINEAR_PATTERN = "LINEAR_PATTERN"
    CIRCULAR_PATTERN = "CIRCULAR_PATTERN"
    # boolean
    BOOLEAN_UNION = "BOOLEAN_UNION"
    BOOLEAN_SUBTRACT = "BOOLEAN_SUBTRACT"
    BOOLEAN_INTERSECT = "BOOLEAN_INTERSECT"


FEATURE_TYPE_NAMES = tuple(t.name for t in FeatureType)


class FeatureOperation(str, enum.Enum):
    ADDITIVE = "additive"
    SUBTRACTIVE = "subtractive"
    NEUTRAL = "neutral"


class Feature:
    """Base class for every parametric CAD feature."""

    type: FeatureType = FeatureType.EXTRUDE
    operation: FeatureOperation = FeatureOperation.ADDITIVE

    def __init__(
        self,
        name: str,
        sketch_ref: str = "",
        params: dict[str, Any] | None = None,
        references: list[str] | None = None,
        description: str = "",
    ) -> None:
        if not name:
            raise ValueError("feature name must be non-empty")
        self.name = name
        self.sketch_ref = sketch_ref
        self.params: dict[str, Any] = dict(params or {})
        self.references: list[str] = list(references or [])
        self.description = description

    # -- validation ----------------------------------------------------------
    def validate(self) -> list[str]:
        """Return a list of validation problems (empty when valid)."""
        problems: list[str] = []
        for key, value in self.params.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value != value:
                problems.append(f"parameter {key!r} is NaN")  # NaN check
        return problems

    # -- serialization ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "operation": self.operation.value,
            "sketch_ref": self.sketch_ref,
            "params": dict(self.params),
            "references": list(self.references),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Feature:
        type_value = str(data["type"])
        feature_cls = FEATURE_REGISTRY.get(type_value)
        if feature_cls is None:
            raise KeyError(f"no feature registered for type {type_value!r}")
        return feature_cls(
            name=str(data["name"]),
            sketch_ref=str(data.get("sketch_ref", "")),
            params=data.get("params") or {},
            references=list(data.get("references") or []),
            description=str(data.get("description", "")),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FEATURE_REGISTRY: dict[str, Callable[..., Feature]] = {}


def register_feature(feature_cls: type[Feature]) -> type[Feature]:
    """Decorator registering a feature class under its type name."""
    FEATURE_REGISTRY[feature_cls.type.value] = feature_cls
    return feature_cls


def create_feature(type_name: str, **kwargs: Any) -> Feature:
    """Instantiate a feature by type name (e.g. ``"EXTRUDE"``)."""
    feature_cls = FEATURE_REGISTRY.get(type_name)
    if feature_cls is None:
        raise KeyError(f"no feature registered for type {type_name!r}")
    return feature_cls(**kwargs)


def known_feature_types() -> list[str]:
    return sorted(FEATURE_REGISTRY)


# ---------------------------------------------------------------------------
# Feature tree & design history
# ---------------------------------------------------------------------------


class FeatureTree:
    """An ordered, dependency-aware collection of features on a part.

    ``order`` preserves the authoring sequence; ``execution_order`` is a
    topological sort respecting ``references`` (parents before children).
    """

    def __init__(self, features: list[Feature] | None = None) -> None:
        self._features: dict[str, Feature] = {}
        self.order: list[str] = []
        for feature in features or []:
            self.add(feature)

    def add(self, feature: Feature) -> Feature:
        if feature.name in self._features:
            raise KeyError(f"feature {feature.name!r} already exists")
        self._features[feature.name] = feature
        self.order.append(feature.name)
        return feature

    def __contains__(self, name: object) -> bool:
        return name in self._features

    def __getitem__(self, name: str) -> Feature:
        return self._features[name]

    def get(self, name: str) -> Feature | None:
        return self._features.get(name)

    def remove(self, name: str) -> Feature:
        feature = self._features.pop(name, None)
        if feature is None:
            raise KeyError(f"feature {name!r} not found")
        self.order = [n for n in self.order if n != name]
        return feature

    def features(self) -> list[Feature]:
        return [self._features[name] for name in self.order]

    def __len__(self) -> int:
        return len(self._features)

    def __iter__(self):
        return iter(self.features())

    def children(self, name: str) -> list[Feature]:
        return [f for f in self.features() if name in f.references]

    def dependencies(self, name: str) -> list[Feature]:
        feature = self._features[name]
        return [self._features[ref] for ref in feature.references if ref in self._features]

    def execution_order(self) -> list[Feature]:
        """Topologically sorted features (parents before children)."""
        visited: set[str] = set()
        result: list[Feature] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            feature = self._features[name]
            for ref in feature.references:
                if ref in self._features:
                    visit(ref)
            result.append(feature)

        for name in self.order:
            visit(name)
        return result

    def to_dict(self) -> list[dict[str, Any]]:
        return [f.to_dict() for f in self.features()]

    @classmethod
    def from_dict(cls, data: list[dict[str, Any]]) -> FeatureTree:
        return cls([Feature.from_dict(item) for item in data])


@dataclass
class HistoryEntry:
    """A single recorded event in a part's design history."""

    event: str  # "add_feature" | "modify_feature" | "remove_feature" | ...
    feature_name: str
    detail: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "feature_name": self.feature_name,
            "detail": self.detail,
            "timestamp": self.timestamp,
        }


class DesignHistory:
    """Append-only log of a part's design events (the *design history*)."""

    def __init__(self, entries: list[HistoryEntry] | None = None) -> None:
        self._entries: list[HistoryEntry] = list(entries or [])

    def record(self, event: str, feature_name: str, **detail: Any) -> HistoryEntry:
        entry = HistoryEntry(event, feature_name, detail)
        self._entries.append(entry)
        return entry

    def entries(self) -> list[HistoryEntry]:
        return list(self._entries)

    def events(self) -> list[str]:
        return [e.event for e in self._entries]

    def feature_events(self, feature_name: str) -> list[HistoryEntry]:
        return [e for e in self._entries if e.feature_name == feature_name]

    def __len__(self) -> int:
        return len(self._entries)

    def to_dict(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._entries]

    @classmethod
    def from_dict(cls, data: list[dict[str, Any]]) -> DesignHistory:
        return cls(
            [
                HistoryEntry(
                    event=str(item["event"]),
                    feature_name=str(item["feature_name"]),
                    detail=item.get("detail") or {},
                    timestamp=str(item.get("timestamp", "")),
                )
                for item in data
            ]
        )


__all__ = [
    "FEATURE_REGISTRY",
    "FEATURE_TYPE_NAMES",
    "DesignHistory",
    "Feature",
    "FeatureOperation",
    "FeatureTree",
    "FeatureType",
    "HistoryEntry",
    "create_feature",
    "known_feature_types",
    "register_feature",
]
