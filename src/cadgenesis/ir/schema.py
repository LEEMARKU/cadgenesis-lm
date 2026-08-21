"""
cadgenesis.ir.schema
====================
CAD-IR vocabulary classification and parameter conventions.

Grounding
---------
The de-facto program format across the repo is a flat list of token strings
(e.g. ``["SKETCH_RECT", "NUM_80", "EXTRUDE", "NUM_10", "BOX"]``) produced by
:mod:`cadgenesis.datasets.cad_program_synth` and consumed by the execution
backends.  This module classifies those tokens into IR roles:

* **base primitives** — ``BOX``, ``CYLINDER``, ``SPHERE``, ``SKETCH_RECT``,
  ``SKETCH`` and the canonical ``PRIM_*`` names;
* **features** — ``EXTRUDE``, ``HOLE``, ``THREAD``, … and canonical ``FEAT_*``;
* **numeric parameters** — ``NUM_*`` / ``ANG_*`` / ``RAT_*``;
* **attributes** — part/quality names such as ``BRACKET`` or ``STEEL``.

Numeric decoding follows the tokenizer convention fixed in M1
(:meth:`cadgenesis.tokenizer.cad_tokenizer.CADTokenizer.decode_length`):

* unpadded ``NUM_<v>`` (fewer than 3 digits) is a **raw millimetre value**;
* zero-padded ``NUM_xxx`` is a **quantizer bin index** decoded through
  :class:`cadgenesis.tokenizer.numeric.NumericTokenizer`.
"""

from __future__ import annotations

from cadgenesis.tokenizer.numeric import NumericTokenizer

#: Version of the CAD-IR schema.  Programs must declare the schema version
#: they were produced with; ``is_schema_compatible`` gates consumption.
CAD_IR_SCHEMA_VERSION = "1.0.0"

#: Canonical kind names for base primitives.
PRIMITIVE_KINDS: frozenset[str] = frozenset(
    {
        "PRIM_BOX",
        "PRIM_CYLINDER",
        "PRIM_SPHERE",
        "PRIM_CONE",
        "PRIM_TORUS",
        "PRIM_WEDGE",
        "PRIM_PRISM",
        "PRIM_PYRAMID",
        "PRIM_ELLIPSOID",
        "PRIM_CAPSULE",
        "PRIM_SOLID",
        "PRIM_SHELL",
        "PRIM_COMPOUND",
    }
)

#: Canonical kind names for feature operations.  The set includes the
#: abstract kinds produced from legacy dataset keywords (``FEAT_PATTERN``,
#: ``FEAT_MIRROR``, ``FEAT_BOOLEAN_UNION``, ``FEAT_BOOLEAN_CUT``,
#: ``FEAT_COUNTERBORE``); the *registered* canonical vocabulary uses the
#: concrete ``FEAT_*`` names (``FEAT_PATTERN_LIN``, ``FEAT_BOOL_UNION``,
#: ``FEAT_HOLE_CB``, ...) which are matched by prefix in
#: :func:`is_feature_kind` / :func:`canonical_kind`.
FEATURE_KINDS: frozenset[str] = frozenset(
    {
        "FEAT_EXTRUDE",
        "FEAT_REVOLVE",
        "FEAT_HOLE",
        "FEAT_THREAD",
        "FEAT_PATTERN",
        "FEAT_FILLET",
        "FEAT_CHAMFER",
        "FEAT_MIRROR",
        "FEAT_BOOLEAN_UNION",
        "FEAT_BOOLEAN_CUT",
        "FEAT_COUNTERBORE",
        "FEAT_SLOT",
    }
)

#: Plain legacy token -> canonical kind (dataset vocabulary, M1).
_LEGACY_KINDS: dict[str, str] = {
    "BOX": "PRIM_BOX",
    "CYLINDER": "PRIM_CYLINDER",
    "SPHERE": "PRIM_SPHERE",
    "SKETCH_RECT": "PRIM_BOX",
    "SKETCH": "PRIM_SOLID",
    "EXTRUDE": "FEAT_EXTRUDE",
    "REVOLVE": "FEAT_REVOLVE",
    "HOLE": "FEAT_HOLE",
    "THREAD": "FEAT_THREAD",
    "PATTERN": "FEAT_PATTERN",
    "FILLET": "FEAT_FILLET",
    "CHAMFER": "FEAT_CHAMFER",
    "MIRROR": "FEAT_MIRROR",
    "BOOLEAN_UNION": "FEAT_BOOLEAN_UNION",
    "BOOLEAN_CUT": "FEAT_BOOLEAN_CUT",
    "COUNTERBORE": "FEAT_COUNTERBORE",
    "SLOT": "FEAT_SLOT",
}

#: Token keywords that open a base primitive op.
BASE_KEYWORDS: frozenset[str] = frozenset(
    k for k, v in _LEGACY_KINDS.items() if v in PRIMITIVE_KINDS
)

#: Token keywords that open a feature op.
FEATURE_KEYWORDS: frozenset[str] = frozenset(_LEGACY_KINDS.keys()) - BASE_KEYWORDS

#: Numeric parameter token prefixes.
NUMERIC_PREFIXES: tuple[str, ...] = ("NUM_", "ANG_", "RAT_")

#: Unpadded names with >= this many digits are canonical zero-padded bins.
_RAW_MM_DIGIT_LIMIT = 3

#: Numeric parameter value range accepted by the validator (millimetres).
PARAM_MIN_MM = 0.0
PARAM_MAX_MM = 1_000.0


def is_base_token(token: str) -> bool:
    """True when ``token`` opens a base primitive op."""
    return token in BASE_KEYWORDS or token in PRIMITIVE_KINDS or token.startswith("PRIM_")


def is_feature_token(token: str) -> bool:
    """True when ``token`` opens a feature op."""
    return token in FEATURE_KEYWORDS or token in FEATURE_KINDS or token.startswith("FEAT_")


def is_numeric_token(token: str) -> bool:
    """True when ``token`` is a numeric parameter token."""
    return token.startswith(NUMERIC_PREFIXES)


def canonical_kind(token: str) -> str:
    """Map any known op keyword to its canonical ``PRIM_*``/``FEAT_*`` kind."""
    if token in _LEGACY_KINDS:
        return _LEGACY_KINDS[token]
    if token in PRIMITIVE_KINDS or token in FEATURE_KINDS:
        return token
    if token.startswith("PRIM_") or token.startswith("FEAT_"):
        return token
    return "RAW"


def is_primitive_kind(kind: str) -> bool:
    return kind in PRIMITIVE_KINDS or kind.startswith("PRIM_")


def is_feature_kind(kind: str) -> bool:
    return kind in FEATURE_KINDS or kind.startswith("FEAT_")


def decode_param_value(token: str) -> float | None:
    """Decode a numeric parameter token to its value in millimetres.

    Unpadded ``NUM_<v>`` (fewer than 3 digits) is a raw millimetre value;
    zero-padded names are quantizer bins (``NUM_*`` lengths, ``ANG_*`` angles,
    ``RAT_*`` ratios).
    """
    if not is_numeric_token(token):
        return None
    if token.startswith("NUM_"):
        digits = token[4:]
        if len(digits) < _RAW_MM_DIGIT_LIMIT:
            try:
                return float(int(digits))
            except ValueError:
                return None
        return NumericTokenizer.decode_length(token)
    if token.startswith("ANG_"):
        return NumericTokenizer.decode_angle(token)
    return NumericTokenizer.decode_ratio(token)


def is_schema_compatible(version: str, current: str = CAD_IR_SCHEMA_VERSION) -> bool:
    """True when ``version`` is consumable by schema ``current``.

    Major version must match; minor and patch may differ (additive changes).
    """
    try:
        cur_major, cur_minor, cur_patch = (int(p) for p in current.split("."))
        major, minor, patch = (int(p) for p in version.split("."))
    except ValueError:
        return False
    if major != cur_major:
        return False
    if minor > cur_minor:
        return False
    return not (minor == cur_minor and patch > cur_patch)


__all__ = [
    "BASE_KEYWORDS",
    "CAD_IR_SCHEMA_VERSION",
    "FEATURE_KEYWORDS",
    "FEATURE_KINDS",
    "NUMERIC_PREFIXES",
    "PARAM_MAX_MM",
    "PARAM_MIN_MM",
    "PRIMITIVE_KINDS",
    "canonical_kind",
    "decode_param_value",
    "is_base_token",
    "is_feature_kind",
    "is_feature_token",
    "is_numeric_token",
    "is_primitive_kind",
    "is_schema_compatible",
]
