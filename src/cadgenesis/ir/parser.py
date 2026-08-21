"""
cadgenesis.ir.parser
====================
Structural parser: flat token program -> :class:`CadProgram`.

Strategy (total, never fails)
-----------------------------
* a base keyword opens a new **primitive** op;
* a feature keyword opens a new **feature** op;
* a numeric token attaches to the current op as a decoded parameter
  (``d0``, ``d1``, … in millimetres);
* any other token attaches to the current op as an **attribute**;
* when no op is open yet, a ``RAW`` op is created on demand.

Dependency edges form a linear chain (each new op depends on its
predecessor) — the only dependency information a flat program carries.
Every input token is assigned to exactly one op in order, so
``parse_program(tokens).to_tokens() == tokens`` always holds.
"""

from __future__ import annotations

from cadgenesis.ir.program import CadOperation, CadProgram, operation_id
from cadgenesis.ir.schema import (
    canonical_kind,
    decode_param_value,
    is_base_token,
    is_feature_token,
    is_numeric_token,
)

_ATTR_KEY = "attr"
_PARAM_PREFIX = "d"


def parse_program(tokens: list[str]) -> CadProgram:
    """Parse a flat token program into a typed :class:`CadProgram`."""
    steps: list[CadOperation] = []
    params: dict[str, object] = {}
    attrs: list[str] = []
    tokens_in_op: list[str] = []
    kind = ""
    position = -1
    param_count = 0

    def flush() -> None:
        nonlocal params, attrs, tokens_in_op, kind, position, param_count
        if not tokens_in_op:
            return
        data: dict[str, object] = dict(params)
        if attrs:
            data[_ATTR_KEY] = list(attrs)
        steps.append(
            CadOperation(
                op_id=operation_id(kind, data, position),
                kind=kind,
                params=data,
                depends_on=(steps[-1].op_id,) if steps else (),
                tokens=tuple(tokens_in_op),
                position=position,
            )
        )
        params = {}
        attrs = []
        tokens_in_op = []
        kind = ""
        position = -1
        param_count = 0

    for i, token in enumerate(tokens):
        if is_base_token(token) or is_feature_token(token):
            flush()
            kind = canonical_kind(token)
            position = i
            tokens_in_op.append(token)
            continue
        if is_numeric_token(token):
            if not tokens_in_op:
                flush()
                kind = "RAW"
                position = i
            tokens_in_op.append(token)
            value = decode_param_value(token)
            if value is not None:
                params[f"{_PARAM_PREFIX}{param_count}"] = round(value, 6)
                param_count += 1
            continue
        # Attribute / part-name token
        if not tokens_in_op:
            flush()
            kind = "RAW"
            position = i
        tokens_in_op.append(token)
        attrs.append(token)
    flush()
    return CadProgram.build(steps)


__all__ = ["parse_program"]
