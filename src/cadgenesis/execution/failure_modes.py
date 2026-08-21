"""
cadgenesis.execution.failure_modes
==================================
Formal failure-mode taxonomy for CAD generation and self-correction.

Failure modes are classified from validation reasons, program structure
and execution results so the self-correction loop (and downstream metrics)
speak a shared, countable vocabulary:

* :class:`FailureMode` — the taxonomy enum.
* :func:`classify_reason` — map a validator reason string to a mode.
* :func:`classify_program` — structural heuristic on a token program.
* :func:`failure_mode_counts` — aggregate counts over results.

This is the classification layer behind ``repair_success_rate`` and
``iterations_to_success``: a repair is only measurable when the failure
it repaired was classified.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class FailureMode(str, Enum):
    """Taxonomy of CAD generation / execution failure modes."""

    MISSING_BASE_SOLID = "missing_base_solid"
    BAD_DIMENSION = "bad_dimension"
    GEOMETRY_INVALID = "geometry_invalid"
    NON_MANIFOLD = "non_manifold"
    UNDERCONSTRAINED = "underconstrained"
    INFEASIBLE_CONSTRAINT = "infeasible_constraint"
    UNKNOWN_TOKEN = "unknown_token"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


#: Base operations that must appear for a structurally valid program.
_BASE_OPS = {"BOX", "CYLINDER", "SPHERE", "SKETCH_RECT", "SKETCH", "PRIM_BOX", "PRIM_CYLINDER"}


def classify_reason(reason: str) -> FailureMode:
    """Map a validator/reason string onto the taxonomy.

    Unknown reasons fall back to :attr:`FailureMode.UNKNOWN` so callers
    never crash on new validator messages.
    """
    if not reason:
        return FailureMode.UNKNOWN
    lowered = reason.lower()
    if "empty token list" in lowered:
        return FailureMode.MISSING_BASE_SOLID
    if "missing base solid" in lowered or "no base" in lowered:
        return FailureMode.MISSING_BASE_SOLID
    if "geometry validation failed" in lowered or "geometry invalid" in lowered:
        return FailureMode.GEOMETRY_INVALID
    if "non-manifold" in lowered or "non_manifold" in lowered:
        return FailureMode.NON_MANIFOLD
    if "underconstrain" in lowered:
        return FailureMode.UNDERCONSTRAINED
    if "infeasible" in lowered or "unsatisfiable" in lowered or "no feasible" in lowered:
        return FailureMode.INFEASIBLE_CONSTRAINT
    if "unknown token" in lowered or "unrecognized token" in lowered or "invalid token" in lowered:
        return FailureMode.UNKNOWN_TOKEN
    if "timeout" in lowered or "timed out" in lowered:
        return FailureMode.TIMEOUT
    if "validator error" in lowered or "execution failed" in lowered or "exception" in lowered:
        return FailureMode.EXECUTION_ERROR
    return FailureMode.UNKNOWN


def classify_program(tokens: list[str]) -> FailureMode:
    """Structural heuristic over a CAD token program.

    Returns ``FailureMode.UNKNOWN`` when no structural defect is found
    (the program may still fail the geometry validator).
    """
    if not tokens:
        return FailureMode.MISSING_BASE_SOLID
    if any(not isinstance(t, str) for t in tokens):
        return FailureMode.EXECUTION_ERROR
    if not any(t in _BASE_OPS for t in tokens):
        return FailureMode.MISSING_BASE_SOLID
    if "EXTRUDE" in tokens:
        idx = tokens.index("EXTRUDE")
        if idx + 1 < len(tokens) and not str(tokens[idx + 1]).startswith("NUM_"):
            return FailureMode.BAD_DIMENSION
    return FailureMode.UNKNOWN


def classify_result(result: Any) -> FailureMode:
    """Classify an execution result (CADExecutionResult or duck-typed).

    Prefers the failure reason embedded in the result, then error strings,
    then the repair report.
    """
    if result is None:
        return FailureMode.UNKNOWN
    reason = getattr(result, "failure_reason", None) or ""
    if isinstance(reason, str) and reason:
        return classify_reason(reason)
    for error in getattr(result, "errors", []) or []:
        mode = classify_reason(str(error))
        if mode is not FailureMode.UNKNOWN:
            return mode
    repair_report = getattr(result, "repair_report", None)
    if isinstance(repair_report, dict):
        message = repair_report.get("message", "") or ""
        if message:
            mode = classify_reason(str(message))
            if mode is not FailureMode.UNKNOWN:
                return mode
    return FailureMode.UNKNOWN


def failure_mode_counts(modes: list[FailureMode]) -> dict[str, int]:
    """Aggregate mode counts; unknown modes are included but never crash."""
    counts: dict[str, int] = {}
    for mode in modes:
        key = mode.value if isinstance(mode, FailureMode) else str(mode)
        counts[key] = counts.get(key, 0) + 1
    return counts


__all__ = [
    "FailureMode",
    "classify_program",
    "classify_reason",
    "classify_result",
    "failure_mode_counts",
]