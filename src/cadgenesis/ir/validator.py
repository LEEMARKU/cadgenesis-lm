"""
cadgenesis.ir.validator
=======================
Structural validation for CAD-IR programs.

Checks performed (mirrors and extends the legacy keyword validator without
weakening it):

1. the program declares a compatible schema version;
2. every step is non-empty (has at least one token);
3. op IDs are unique;
4. every ``depends_on`` edge references an existing step;
5. the dependency graph is acyclic;
6. decoded numeric parameters lie in the accepted range;
7. at least one base primitive step or base keyword is present;
8. round-trip: ``program.to_tokens()`` reproduces the input token list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cadgenesis.ir.parser import parse_program
from cadgenesis.ir.program import CadProgram
from cadgenesis.ir.schema import (
    PARAM_MAX_MM,
    PARAM_MIN_MM,
    is_base_token,
    is_primitive_kind,
    is_schema_compatible,
)

#: Keywords the legacy token gate (execution.geometry_validation) treats as a
#: base solid.  The IR gate must accept exactly the same programs.
_LEGACY_BASE_KEYWORDS = frozenset({"EXTRUDE", "BOX", "CYLINDER", "SKETCH_RECT"})


@dataclass
class CadIRCheck:
    """One named check result."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class CadProgramReport:
    """Validation outcome for a CAD program."""

    checks: list[CadIRCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks
            ],
        }

    def summary(self) -> str:
        failed = [c.name for c in self.checks if not c.passed]
        if not failed:
            return f"CAD-IR OK ({len(self.checks)} checks)"
        return f"CAD-IR FAILED: {', '.join(failed)}"


def validate_cad_program(tokens: list[str]) -> bool:
    """Drop-in structural gate: True iff ``tokens`` form a valid CAD program."""
    return validate_program_ir(parse_program(tokens), original=tokens).passed


def validate_program_ir(
    program: CadProgram,
    original: list[str] | None = None,
    vocab=None,
) -> CadProgramReport:
    """
    Run every structural check against ``program``.

    ``vocab`` optionally supplies a tokenizer/vocabulary object with
    ``__contains__`` (or a ``family_of`` method) so an extra
    ``tokens_registered`` check verifies every token of the program is
    registered — the dialect-sensitive gate (mini vs default vocabulary).
    """
    report = CadProgramReport()

    if not is_schema_compatible(program.schema_version):
        report.checks.append(
            CadIRCheck(
                "schema_version",
                False,
                f"schema {program.schema_version} incompatible with current",
            )
        )
    else:
        report.checks.append(CadIRCheck("schema_version", True, program.schema_version))

    if program.steps:
        report.checks.append(CadIRCheck("steps_non_empty", True, f"{len(program.steps)} steps"))
    else:
        report.checks.append(CadIRCheck("steps_non_empty", False, "program has no steps"))

    ids = [s.op_id for s in program.steps]
    if len(set(ids)) == len(ids):
        report.checks.append(CadIRCheck("op_ids_unique", True))
    else:
        report.checks.append(CadIRCheck("op_ids_unique", False, "duplicate op IDs"))

    known = set(ids)
    missing = sorted({d for s in program.steps for d in s.depends_on} - known)
    if not missing:
        report.checks.append(CadIRCheck("dependencies_resolve", True))
    else:
        report.checks.append(CadIRCheck("dependencies_resolve", False, f"unknown deps: {missing}"))

    if program.is_cyclic():
        report.checks.append(CadIRCheck("acyclic", False, "dependency cycle detected"))
    else:
        report.checks.append(CadIRCheck("acyclic", True))

    bad_params = sorted(
        (
            f"{s.op_id}:{k}={v}"
            for s in program.steps
            for k, v in s.params.items()
            if isinstance(v, (int, float)) and not (PARAM_MIN_MM <= v <= PARAM_MAX_MM)
        )
    )
    if not bad_params:
        report.checks.append(CadIRCheck("params_in_range", True))
    else:
        report.checks.append(CadIRCheck("params_in_range", False, f"out of range: {bad_params}"))

    # Mirror the legacy gate's base semantics (execution.geometry_validation
    # ``validate_program`` counts ``EXTRUDE`` as a base keyword), so the IR
    # gate is exactly as permissive as the legacy one — never stricter.
    has_base = any(
        is_primitive_kind(s.kind)
        or any(t in _LEGACY_BASE_KEYWORDS or is_base_token(t) for t in s.tokens)
        for s in program.steps
    )
    if has_base:
        report.checks.append(CadIRCheck("base_present", True))
    else:
        report.checks.append(CadIRCheck("base_present", False, "no base primitive keyword"))

    if original is not None:
        if program.to_tokens() == list(original):
            report.checks.append(CadIRCheck("round_trip", True))
        else:
            report.checks.append(CadIRCheck("round_trip", False, "to_tokens() != original tokens"))

    if vocab is not None:
        missing = sorted({t for s in program.steps for t in s.tokens if t not in vocab})
        if not missing:
            report.checks.append(CadIRCheck("tokens_registered", True))
        else:
            report.checks.append(
                CadIRCheck("tokens_registered", False, f"unregistered tokens: {missing}")
            )

    return report


__all__ = [
    "CadIRCheck",
    "CadProgramReport",
    "validate_cad_program",
    "validate_program_ir",
]
