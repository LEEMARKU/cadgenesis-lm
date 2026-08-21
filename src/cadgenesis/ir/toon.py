"""
cadgenesis.ir.toon
==================
TOON bridge for the CAD IR (v6.3).

Mirrors the de-facto TOON object grammar used across the repo
(:mod:`cadgenesis.distillation.synthetic`, ``sdk/toon_extended.py``):

* row fields ``id|feature|width|height|depth|fillet`` with the typed schema
  line ``int|str|float|float|float|float``;
* features ``BOX``, ``CYLINDER``, ``SPHERE``, ``EXTRUDE_PROFILE``.

:func:`program_to_toon` serializes a :class:`~cadgenesis.ir.program.CadProgram`
into that grammar (unmappable ops are reported, never silently dropped);
:func:`toon_to_program` parses TOON rows back into a typed program.  The
pair round-trips losslessly for TOON-shaped programs:

    ``toon_to_program(toon).to_toon() == toon``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cadgenesis.ir.program import CadOperation, CadProgram, operation_id
from sdk import toon_extended

#: TOON object-row fields (order defines the header line).
TOON_FIELDS: tuple[str, ...] = ("id", "feature", "width", "height", "depth", "fillet")

#: Typed schema line written with ``include_schema=True``.
TOON_TYPES: tuple[str, ...] = ("int", "str", "float", "float", "float", "float")

#: Features the TOON grammar knows.
TOON_FEATURES: tuple[str, ...] = ("BOX", "CYLINDER", "SPHERE", "EXTRUDE_PROFILE")

#: IR kind -> TOON feature.
_KIND_TO_TOON: dict[str, str] = {
    "PRIM_BOX": "BOX",
    "PRIM_CYLINDER": "CYLINDER",
    "PRIM_SPHERE": "SPHERE",
    "FEAT_EXTRUDE": "EXTRUDE_PROFILE",
}

#: TOON feature -> IR kind.
_TOON_TO_KIND: dict[str, str] = {v: k for k, v in _KIND_TO_TOON.items()}

#: Params consumed from op ``params`` when serializing.
_DIM_KEYS: tuple[str, ...] = ("width", "height", "depth")


@dataclass
class ToonProgramReport:
    """Result of a program<->TOON conversion (what was mapped, what was not)."""

    toon: str = ""
    skipped: list[str] = field(default_factory=list)

    @property
    def fully_mapped(self) -> bool:
        return not self.skipped


def program_to_toon(program: CadProgram, *, include_schema: bool = True) -> ToonProgramReport:
    """
    Serialize ``program`` into the TOON object grammar.

    Every step whose kind maps onto a TOON feature becomes one object row
    (width/height/depth from the op's decoded ``d0..d2`` params, fillet from
    ``params["fillet"]``).  Ops without a TOON mapping are listed in
    ``report.skipped`` — they are never silently dropped.
    """
    objects: list[dict[str, Any]] = []
    skipped: list[str] = []
    for i, step in enumerate(program.steps):
        feature = _KIND_TO_TOON.get(step.kind)
        if feature is None:
            skipped.append(step.kind)
            continue
        params = step.params
        row: dict[str, Any] = {"id": i + 1, "feature": feature}
        for key in _DIM_KEYS:
            row[key] = _param_or_none(params, f"d{_DIM_KEYS.index(key)}")
        row["fillet"] = params.get("fillet")
        objects.append(row)
    toon = toon_extended.to_toon(
        objects,
        include_schema=include_schema,
        types=list(TOON_TYPES) if include_schema else None,
    )
    return ToonProgramReport(toon=toon, skipped=skipped)


def toon_to_program(toon_str: str) -> CadProgram:
    """
    Parse a TOON payload into a typed :class:`CadProgram`.

    Each object row becomes one operation whose kind is the TOON feature's IR
    kind; ``width/height/depth`` are stored as decoded params ``d0/d1/d2``
    (millimetres) and ``fillet`` under ``params["fillet"]``.  Each op's
    ``tokens`` hold the feature name (the truthful token source of a TOON
    row); program/op IDs are deterministic content hashes.
    """
    if not toon_str.strip():
        return CadProgram.build([])
    objects = toon_extended.from_toon(toon_str)
    steps: list[CadOperation] = []
    for i, obj in enumerate(objects):
        feature = str(obj.get("feature", "")).strip()
        kind = _TOON_TO_KIND.get(feature, "RAW")
        params: dict[str, Any] = {}
        for key in _DIM_KEYS:
            value = _as_float(obj.get(key))
            if value is not None:
                params[f"d{_DIM_KEYS.index(key)}"] = value
        fillet = _as_float(obj.get("fillet"))
        if fillet is not None:
            params["fillet"] = fillet
        steps.append(
            CadOperation(
                op_id=operation_id(kind, params, i),
                kind=kind,
                params=params,
                depends_on=(steps[-1].op_id,) if steps else (),
                tokens=(feature,),
                position=i,
            )
        )
    return CadProgram.build(steps)


def _param_or_none(params: dict[str, Any], key: str) -> float | None:
    value = params.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "TOON_FEATURES",
    "TOON_FIELDS",
    "TOON_TYPES",
    "ToonProgramReport",
    "program_to_toon",
    "toon_to_program",
]