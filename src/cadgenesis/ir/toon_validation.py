"""
cadgenesis.ir.toon_validation
=============================
Semantic validation for TOON payloads (v6.3), with rule parity to the
distillation critique engine (:mod:`cadgenesis.distillation.critique`) and
the tokenizer sequence gate (:mod:`cadgenesis.tokenizer.validation`).

Checks (each one named check in a :class:`~cadgenesis.ir.validator.CadProgramReport`):

1. ``toon_parse`` — the payload parses to at least one object row;
2. ``toon_features`` — every row has a non-empty, known feature token;
3. ``toon_dims_numeric`` — dimensional fields parse as numbers;
4. ``toon_dims_positive`` — width/height/depth/radius/diameter/thickness are
   strictly positive when present (mirrors the critique engine);
5. ``toon_fillet_ratio`` — fillet is non-negative and no larger than
   ``fillet_max_ratio`` x the smallest of width/height/depth (default 0.5,
   matching the critique engine).
"""

from __future__ import annotations

from dataclasses import dataclass

from cadgenesis.ir.toon import TOON_FEATURES
from cadgenesis.ir.validator import CadIRCheck, CadProgramReport
from sdk import toon_extended

#: Dimension keys that must be strictly positive when present.  Deliberately
#: identical to ``distillation.critique._POSITIVE_DIMENSION_KEYS`` (parity is
#: enforced by tests/ir/test_toon.py); defined locally to avoid importing the
#: whole distillation stack from the IR package (circular import).
POSITIVE_DIMENSION_KEYS: tuple[str, ...] = (
    "width",
    "height",
    "depth",
    "radius",
    "diameter",
    "thickness",
)


@dataclass
class ToonValidationConfig:
    """Tunables mirroring the critique engine defaults."""

    fillet_max_ratio: float = 0.5
    positive_dimension_keys: tuple[str, ...] = POSITIVE_DIMENSION_KEYS


def validate_toon_program(
    toon_str: str,
    config: ToonValidationConfig | None = None,
) -> CadProgramReport:
    """
    Validate a TOON payload against the CAD object grammar + critique rules.
    """
    cfg = config or ToonValidationConfig()
    report = CadProgramReport()

    try:
        objects = toon_extended.from_toon(toon_str)
    except Exception as exc:
        report.checks.append(CadIRCheck("toon_parse", False, f"parse error: {exc}"))
        report.checks.append(CadIRCheck("toon_features", False, "payload unparsable"))
        report.checks.append(CadIRCheck("toon_dims_numeric", False, "payload unparsable"))
        report.checks.append(CadIRCheck("toon_dims_positive", False, "payload unparsable"))
        report.checks.append(CadIRCheck("toon_fillet_ratio", False, "payload unparsable"))
        return report

    if objects:
        report.checks.append(CadIRCheck("toon_parse", True, f"{len(objects)} rows"))
    else:
        report.checks.append(
            CadIRCheck("toon_parse", False, "empty or unparsable TOON payload")
        )

    # --- features ---------------------------------------------------------
    empty = [i for i, obj in enumerate(objects) if not str(obj.get("feature", "")).strip()]
    unknown = [
        i
        for i, obj in enumerate(objects)
        if str(obj.get("feature", "")).strip() not in TOON_FEATURES
    ]
    if empty:
        report.checks.append(
            CadIRCheck("toon_features", False, f"rows with empty feature: {empty}")
        )
    elif unknown:
        report.checks.append(
            CadIRCheck(
                "toon_features",
                False,
                f"unknown feature tokens on rows {unknown}; known: {list(TOON_FEATURES)}",
            )
        )
    else:
        report.checks.append(CadIRCheck("toon_features", True, f"{len(objects)} features"))

    # --- dimensions -------------------------------------------------------
    non_numeric: list[str] = []
    non_positive: list[str] = []
    for i, obj in enumerate(objects):
        for key in cfg.positive_dimension_keys:
            value = obj.get(key)
            if value is None or value == "":
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                non_numeric.append(f"row {i} {key}={value!r}")
                continue
            if numeric <= 0:
                non_positive.append(f"row {i} {key}={numeric}")
    if non_numeric:
        report.checks.append(
            CadIRCheck("toon_dims_numeric", False, f"non-numeric: {non_numeric}")
        )
    else:
        report.checks.append(CadIRCheck("toon_dims_numeric", True))

    if non_positive:
        report.checks.append(
            CadIRCheck("toon_dims_positive", False, f"non-positive: {non_positive}")
        )
    else:
        report.checks.append(CadIRCheck("toon_dims_positive", True))

    # --- fillet ratio -----------------------------------------------------
    bad_fillets: list[str] = []
    for i, obj in enumerate(objects):
        fillet = obj.get("fillet")
        if fillet is None or fillet == "":
            continue
        try:
            fillet_value = float(fillet)
        except (TypeError, ValueError):
            bad_fillets.append(f"row {i} fillet={fillet!r}")
            continue
        dims = []
        for key in ("width", "height", "depth"):
            value = obj.get(key)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric > 0:
                dims.append(numeric)
        if fillet_value < 0:
            bad_fillets.append(f"row {i} negative fillet={fillet_value}")
        elif dims and fillet_value > cfg.fillet_max_ratio * min(dims):
            bad_fillets.append(
                f"row {i} fillet={fillet_value} exceeds "
                f"{cfg.fillet_max_ratio} x min({min(dims):.2f})"
            )
    if bad_fillets:
        report.checks.append(CadIRCheck("toon_fillet_ratio", False, "; ".join(bad_fillets)))
    else:
        report.checks.append(CadIRCheck("toon_fillet_ratio", True))

    return report


def toon_program_is_valid(toon_str: str, **kwargs) -> bool:
    """Drop-in gate: True iff ``toon_str`` passes all TOON semantic checks."""
    return validate_toon_program(toon_str, **kwargs).passed


__all__ = [
    "ToonValidationConfig",
    "toon_program_is_valid",
    "validate_toon_program",
]