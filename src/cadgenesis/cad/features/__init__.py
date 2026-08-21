"""cadgenesis.cad.features
========================
Feature-based modelling: the full parametric feature-operation vocabulary
(extrude, revolve, loft, sweep, fillet, chamfer, shell, draft, hole, rib,
mirror, pattern, boolean...), the feature tree and the design history.

Importing this package registers every built-in feature in
:data:`cadgenesis.cad.features.base.FEATURE_REGISTRY`.
"""

from cadgenesis.cad.features.base import (
    FEATURE_REGISTRY,
    FEATURE_TYPE_NAMES,
    DesignHistory,
    Feature,
    FeatureOperation,
    FeatureTree,
    FeatureType,
    HistoryEntry,
    create_feature,
    known_feature_types,
    register_feature,
)
from cadgenesis.cad.features.boolean import (
    BooleanIntersect,
    BooleanSubtract,
    BooleanUnion,
)
from cadgenesis.cad.features.dress import Chamfer, Draft, Fillet
from cadgenesis.cad.features.patterns import (
    CircularPattern,
    LinearPattern,
    Mirror,
)
from cadgenesis.cad.features.solids import (
    Cut,
    Extrude,
    Hole,
    Loft,
    Pocket,
    Revolve,
    Rib,
    Shell,
    Sweep,
    Thicken,
)

__all__ = [
    "FEATURE_REGISTRY",
    "FEATURE_TYPE_NAMES",
    "BooleanIntersect",
    "BooleanSubtract",
    "BooleanUnion",
    "Chamfer",
    "CircularPattern",
    "Cut",
    "DesignHistory",
    "Draft",
    "Extrude",
    "Feature",
    "FeatureOperation",
    "FeatureTree",
    "FeatureType",
    "Fillet",
    "HistoryEntry",
    "Hole",
    "LinearPattern",
    "Loft",
    "Mirror",
    "Pocket",
    "Revolve",
    "Rib",
    "Shell",
    "Sweep",
    "Thicken",
    "create_feature",
    "known_feature_types",
    "register_feature",
]
