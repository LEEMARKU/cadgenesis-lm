"""
cadgenesis.ir
=============
Typed CAD intermediate representation.

The IR gives every flat token program a typed, versioned, dependency-ordered
structure (see :mod:`cadgenesis.ir.schema` for the grounding conventions).
The parse -> validate -> serialize pipeline is lossless:

    parse_program(tokens).to_tokens() == tokens

and deterministic content IDs make programs dedup- and cache-friendly.
"""

from cadgenesis.ir.diff import IrDiffReport, ir_diff
from cadgenesis.ir.parser import parse_program
from cadgenesis.ir.program import CadOperation, CadProgram, operation_id
from cadgenesis.ir.schema import (
    BASE_KEYWORDS,
    CAD_IR_SCHEMA_VERSION,
    FEATURE_KEYWORDS,
    FEATURE_KINDS,
    NUMERIC_PREFIXES,
    PRIMITIVE_KINDS,
    canonical_kind,
    decode_param_value,
    is_base_token,
    is_feature_kind,
    is_feature_token,
    is_numeric_token,
    is_primitive_kind,
    is_schema_compatible,
)
from cadgenesis.ir.toon import (
    TOON_FEATURES,
    TOON_FIELDS,
    TOON_TYPES,
    ToonProgramReport,
    program_to_toon,
    toon_to_program,
)
from cadgenesis.ir.toon_validation import (
    ToonValidationConfig,
    toon_program_is_valid,
    validate_toon_program,
)
from cadgenesis.ir.validator import (
    CadIRCheck,
    CadProgramReport,
    validate_cad_program,
    validate_program_ir,
)

__all__ = [
    "BASE_KEYWORDS",
    "CAD_IR_SCHEMA_VERSION",
    "FEATURE_KEYWORDS",
    "FEATURE_KINDS",
    "NUMERIC_PREFIXES",
    "PRIMITIVE_KINDS",
    "TOON_FEATURES",
    "TOON_FIELDS",
    "TOON_TYPES",
    "CadIRCheck",
    "CadOperation",
    "CadProgram",
    "CadProgramReport",
    "IrDiffReport",
    "ToonProgramReport",
    "ToonValidationConfig",
    "canonical_kind",
    "decode_param_value",
    "ir_diff",
    "is_base_token",
    "is_feature_kind",
    "is_feature_token",
    "is_numeric_token",
    "is_primitive_kind",
    "is_schema_compatible",
    "operation_id",
    "parse_program",
    "program_to_toon",
    "toon_program_is_valid",
    "toon_to_program",
    "validate_cad_program",
    "validate_program_ir",
    "validate_toon_program",
]
