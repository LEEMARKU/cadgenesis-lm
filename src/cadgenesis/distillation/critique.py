"""cadgenesis.distillation.critique
================================
Critique-based self-improvement loop.

A deterministic, rule-based critiquer over TOON strings that powers the
self-improvement loop and feeds AI-feedback (RLAIF) preference signals.
No LLM calls are made: everything is derived from parsing the TOON payload
via ``sdk.toon_extended.from_toon`` plus simple keyword matching against the
prompt.

Checks performed by :meth:`CritiqueEngine.critique`
---------------------------------------------------
1. **Parseability** -- the TOON payload must parse and contain objects.
2. **Non-empty features** -- every object row must carry a feature token.
3. **Positive dimensions** -- ``width``/``height``/``depth``/``radius``/
   ``diameter``/``thickness`` must be strictly positive.
4. **Expected feature presence** -- if the prompt mentions a primitive
   keyword (box, cylinder, sphere, extrude), at least one object must use
   that feature (simple case-insensitive keyword matching).
5. **Fillet sanity** -- ``fillet`` must be non-negative and no larger than
   ``fillet_max_ratio`` times the smallest of width/height/depth.

Score
-----
The score starts at ``1.0`` and loses ``0.25`` per issue, clamped to
``[0, 1]`` (two decimals).  ``build_feedback_from_errors`` produces the
same scoring from an explicit issue list, so callers can extend the rule
set with their own checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sdk import toon_extended

__all__ = ["CritiqueEngine", "CritiqueFeedback"]

#: prompt keyword -> expected TOON feature token (disjoint keyword sets).
FEATURE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "BOX": ("box", "block", "bracket", "enclosure", "plate", "mounting", "base"),
    "CYLINDER": ("cylinder", "cylindrical", "shaft", "adapter", "rod", "pin", "pipe", "sleeve"),
    "SPHERE": ("sphere", "spherical", "ball", "knob"),
    "EXTRUDE_PROFILE": ("extrude", "extruded", "profile", "beam", "cross section", "i-profile"),
}

#: dimension keys that must be strictly positive.
_POSITIVE_DIMENSION_KEYS: tuple[str, ...] = (
    "width",
    "height",
    "depth",
    "radius",
    "diameter",
    "thickness",
)

_ISSUE_SUGGESTIONS: dict[str, str] = {
    "parse": "Regenerate the TOON payload with valid syntax and a schema header.",
    "empty": "Ensure at least one object row is present with a non-empty feature token.",
    "dimension": "Use strictly positive values for all dimensional parameters.",
    "feature": (
        "Include the primitive feature (BOX/CYLINDER/SPHERE/EXTRUDE_PROFILE) "
        "referenced by the prompt."
    ),
    "fillet": "Keep fillet non-negative and no larger than half the smallest dimension.",
}


@dataclass
class CritiqueFeedback:
    """Result of critiquing one TOON string.

    ``score`` is in ``[0, 1]`` (higher = better); ``issues`` lists human
    readable problems found; ``suggestions`` are the corresponding fixes.
    """

    score: float
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


class CritiqueEngine:
    """Rule-based critiquer over TOON strings (no LLM calls)."""

    def __init__(self, fillet_max_ratio: float = 0.5, issue_penalty: float = 0.25) -> None:
        if not 0.0 < fillet_max_ratio <= 1.0:
            raise ValueError(f"fillet_max_ratio must be in (0, 1], got {fillet_max_ratio}")
        if not 0.0 < issue_penalty <= 1.0:
            raise ValueError(f"issue_penalty must be in (0, 1], got {issue_penalty}")
        self.fillet_max_ratio = fillet_max_ratio
        self.issue_penalty = issue_penalty

    def critique(self, toon: str, prompt: str) -> CritiqueFeedback:
        """Critique ``toon`` given the original ``prompt``."""
        errors: list[str] = []
        try:
            objects = toon_extended.from_toon(toon)
        except Exception as exc:
            return self.build_feedback_from_errors(toon, [f"TOON parse error: {exc}"])

        if not objects:
            feedback = self.build_feedback_from_errors(toon, ["Empty or unparsable TOON payload."])
            feedback.score = 0.0  # an unparsable payload is a total failure
            return feedback

        self._check_features(errors, objects)
        self._check_dimensions(errors, objects)
        self._check_fillets(errors, objects)
        self._check_expected_features(errors, prompt, objects)

        return self.build_feedback_from_errors(toon, errors)

    def build_feedback_from_errors(self, toon: str, errors: list[str]) -> CritiqueFeedback:
        """Build a :class:`CritiqueFeedback` from an explicit issue list."""
        score = max(0.0, 1.0 - self.issue_penalty * len(errors))
        suggestions = []
        for error in errors:
            error_lower = error.lower()
            matched = next(
                (hint for key, hint in _ISSUE_SUGGESTIONS.items() if key in error_lower),
                f"Review and correct: {error}",
            )
            suggestions.append(matched)
        return CritiqueFeedback(score=round(score, 2), issues=list(errors), suggestions=suggestions)

    # ------------------------------------------------------------- checks

    @staticmethod
    def _check_features(errors: list[str], objects: list[dict]) -> None:
        for i, obj in enumerate(objects):
            feature = str(obj.get("feature", "")).strip()
            if not feature:
                errors.append(f"Object {i}: empty feature token (non-empty feature required).")

    @staticmethod
    def _check_dimensions(errors: list[str], objects: list[dict]) -> None:
        for i, obj in enumerate(objects):
            for key in _POSITIVE_DIMENSION_KEYS:
                value = obj.get(key)
                if value is None or value == "":
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    errors.append(f"Object {i}: non-numeric '{key}' value ({value}).")
                    continue
                if numeric <= 0:
                    errors.append(f"Object {i}: '{key}' must be positive, got {numeric}.")

    def _check_fillets(self, errors: list[str], objects: list[dict]) -> None:
        for i, obj in enumerate(objects):
            fillet = obj.get("fillet")
            if fillet is None or fillet == "":
                continue
            try:
                fillet_numeric = float(fillet)
            except (TypeError, ValueError):
                errors.append(f"Object {i}: non-numeric 'fillet' value ({fillet}).")
                continue
            if fillet_numeric < 0:
                errors.append(f"Object {i}: 'fillet' must be non-negative, got {fillet_numeric}.")
                continue
            smallest = None
            for key in ("width", "height", "depth"):
                value = obj.get(key)
                if value is None or value == "":
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                smallest = numeric if smallest is None else min(smallest, numeric)
            if smallest is not None and fillet_numeric > self.fillet_max_ratio * smallest:
                errors.append(
                    f"Object {i}: 'fillet' {fillet_numeric} exceeds "
                    f"{self.fillet_max_ratio} of the smallest dimension ({smallest})."
                )

    def _check_expected_features(self, errors: list[str], prompt: str, objects: list[dict]) -> None:
        prompt_lower = prompt.lower()
        present = {str(obj.get("feature", "")).strip() for obj in objects}
        for feature, keywords in FEATURE_KEYWORDS.items():
            if any(keyword in prompt_lower for keyword in keywords) and feature not in present:
                errors.append(
                    f"Prompt references '{keywords[0]}' but no object uses the '{feature}' feature."
                )
