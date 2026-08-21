"""cadgenesis.world_model.affordances
====================================
Affordance mapping (Pillar 4).

:class:`AffordanceMapper` answers *"what can you do with this part?"* —
the interaction possibilities a geometry invites (grip, rotate, slide,
insert, mate, roll, access, cut, mount).  Mappings are heuristic scores
derived from the primitive feature, its parameters and its pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cadgenesis.world_model.objects import WorldObject

AFFORDANCE_ACTIONS = (
    "grip",
    "rotate",
    "slide",
    "insert",
    "mate",
    "roll",
    "access",
    "cut",
    "mount",
    "press",
)

# feature -> base affordance scores (0..1)
_FEATURE_AFFORDANCES: dict[str, dict[str, float]] = {
    "block": {"grip": 0.6, "slide": 0.5, "mate": 0.7, "mount": 0.7, "press": 0.4},
    "cylinder": {"grip": 0.8, "rotate": 0.9, "slide": 0.7, "mate": 0.8, "roll": 0.9, "mount": 0.5},
    "sphere": {"grip": 0.5, "roll": 1.0, "rotate": 0.8, "mate": 0.5},
    "cone": {"grip": 0.6, "rotate": 0.6, "insert": 0.5, "press": 0.6},
    "torus": {"grip": 0.7, "rotate": 0.6, "access": 0.7, "mate": 0.6},
    "prism": {"grip": 0.5, "slide": 0.6, "mate": 0.7},
    "hole": {"insert": 1.0, "access": 0.9, "mate": 0.8},
    "fillet": {"access": 0.4},
    "chamfer": {"access": 0.5, "insert": 0.4},
    "extrusion": {"slide": 0.6, "mount": 0.6},
    "revolve": {"rotate": 0.9, "grip": 0.7, "mount": 0.5},
    "loft": {"grip": 0.5, "mate": 0.5},
}


@dataclass
class Affordance:
    """A single mapped interaction capability."""

    action: str
    score: float
    basis: str = "feature"

    def summary(self) -> dict[str, Any]:
        return {"action": self.action, "score": round(self.score, 3), "basis": self.basis}


class AffordanceMapper:
    """Map world objects to interaction affordances."""

    def __init__(self, threshold: float = 0.3) -> None:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold

    def affordances(self, obj: WorldObject) -> list[Affordance]:
        """All affordances for an object above the score threshold, sorted desc."""
        base = _FEATURE_AFFORDANCES.get(obj.feature, {})
        scored: list[Affordance] = []
        for action, score in base.items():
            if score >= self.threshold:
                scored.append(Affordance(action, round(score, 3)))
        scored.sort(key=lambda a: a.score, reverse=True)
        return scored

    def supports(self, obj: WorldObject, action: str) -> Affordance | None:
        """Best-matching affordance for ``action`` or None if below threshold."""
        best: Affordance | None = None
        for a in self.affordances(obj):
            if a.action == action and (best is None or a.score > best.score):
                best = a
        return best

    def supports_any(self, obj: WorldObject, actions: list[str]) -> list[str]:
        """Subset of ``actions`` the object supports."""
        supported = {a.action for a in self.affordances(obj)}
        return [a for a in actions if a in supported]


__all__ = ["AFFORDANCE_ACTIONS", "Affordance", "AffordanceMapper"]
