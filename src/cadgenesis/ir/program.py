"""
cadgenesis.ir.program
=====================
Typed CAD program IR: operations, dependency edges, IDs, serialization.

A :class:`CadProgram` is a structured, versioned representation of a flat
token program.  Every operation carries:

* ``op_id`` — deterministic content hash (stable across runs, dedup-friendly);
* ``kind`` — canonical ``PRIM_*`` / ``FEAT_*`` kind (or ``RAW``);
* ``params`` — decoded numeric parameters (mm) plus attribute tokens;
* ``tokens`` — the exact original token slice (enables lossless round-trip);
* ``depends_on`` — dependency edges forming a DAG;
* ``position`` — original index in the flat program.

The IR is lossless: ``parse_program(tokens).to_tokens() == tokens``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from cadgenesis.ir.schema import CAD_IR_SCHEMA_VERSION

#: Namespace separator for dependency/ID prefixing.
_ID_PREFIX = "cadop"


@dataclass(frozen=True)
class CadOperation:
    """One typed operation in a CAD program."""

    op_id: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    tokens: tuple[str, ...] = ()
    position: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_id": self.op_id,
            "kind": self.kind,
            "params": self.params,
            "depends_on": list(self.depends_on),
            "tokens": list(self.tokens),
            "position": self.position,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CadOperation:
        return cls(
            op_id=str(data["op_id"]),
            kind=str(data["kind"]),
            params=dict(data.get("params", {})),
            depends_on=tuple(str(d) for d in data.get("depends_on", [])),
            tokens=tuple(str(t) for t in data.get("tokens", [])),
            position=int(data.get("position", -1)),
        )


@dataclass(frozen=True)
class CadProgram:
    """A versioned, dependency-ordered CAD program."""

    program_id: str
    schema_version: str
    steps: tuple[CadOperation, ...] = ()

    # ------------------------------------------------------------ structure

    def to_tokens(self) -> list[str]:
        """Reconstruct the exact flat token program (lossless round-trip)."""
        return [t for step in self.steps for t in step.tokens]

    def topological_order(self) -> list[str]:
        """Kahn topological order of step ids (stable, deterministic)."""
        ids = [s.op_id for s in self.steps]
        index = {oid: i for i, oid in enumerate(ids)}
        deps: dict[str, set[str]] = {oid: set() for oid in ids}
        for step in self.steps:
            for dep in step.depends_on:
                if dep in index:
                    deps[step.op_id].add(dep)
        order: list[str] = []
        remaining = set(ids)
        while remaining:
            ready = sorted(oid for oid in remaining if deps[oid] <= set(order))
            if not ready:
                break
            for oid in ready:
                order.append(oid)
                remaining.discard(oid)
        order.extend(sorted(remaining))
        return order

    def is_cyclic(self) -> bool:
        return len(self.topological_order()) < len(self.steps)

    # ------------------------------------------------------------ serialization

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "schema_version": self.schema_version,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CadProgram:
        return cls(
            program_id=str(data["program_id"]),
            schema_version=str(data.get("schema_version", CAD_IR_SCHEMA_VERSION)),
            steps=tuple(CadOperation.from_dict(s) for s in data.get("steps", [])),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> CadProgram:
        return cls.from_dict(json.loads(text))

    # ------------------------------------------------------------ factories

    @classmethod
    def build(
        cls,
        steps: list[CadOperation],
        schema_version: str = CAD_IR_SCHEMA_VERSION,
    ) -> CadProgram:
        """Build a program, computing its deterministic content-hash ID."""
        ordered = tuple(steps)
        payload = json.dumps(
            {"schema_version": schema_version, "steps": [s.to_dict() for s in ordered]},
            sort_keys=True,
            ensure_ascii=False,
        )
        program_id = _content_id("program", payload)
        return cls(program_id=program_id, schema_version=schema_version, steps=ordered)


def _content_id(prefix: str, payload: str) -> str:
    """Deterministic, collision-resistant content ID (first 12 hex chars)."""
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest[:12]}"


def operation_id(kind: str, params: dict[str, Any], position: int) -> str:
    """Deterministic op ID from kind + params + position."""
    payload = json.dumps(
        {"kind": kind, "params": params, "position": position},
        sort_keys=True,
        ensure_ascii=False,
    )
    return _content_id(_ID_PREFIX, payload)


__all__ = [
    "CadOperation",
    "CadProgram",
    "operation_id",
]
