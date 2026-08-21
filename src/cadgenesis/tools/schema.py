"""cadgenesis.tools.schema
========================
Tool-calling schema primitives: permissions, parameter specs, tool
definitions and call envelopes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

#: Supported parameter value types.
PARAM_TYPES = ("string", "number", "boolean", "list", "program")


class Permission(str, Enum):
    """Tool permission levels, ordered from least to most privileged."""

    READ = "read"
    EXECUTE = "execute"
    ADMIN = "admin"


_PERMISSION_ORDER = {Permission.READ: 0, Permission.EXECUTE: 1, Permission.ADMIN: 2}


def permission_allows(required: Permission, granted: Permission) -> bool:
    """True when ``granted`` is at least as privileged as ``required``."""
    return _PERMISSION_ORDER[granted] >= _PERMISSION_ORDER[required]


@dataclass(frozen=True)
class ParameterSpec:
    """Declaration of a single tool-call argument."""

    name: str
    type: str
    required: bool = True
    default: Any = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.type not in PARAM_TYPES:
            raise ValueError(
                f"parameter {self.name!r}: type must be one of {PARAM_TYPES}, got {self.type!r}"
            )


@dataclass(frozen=True)
class ToolDefinition:
    """A callable tool with schema, permission level and handler."""

    name: str
    description: str
    parameters: tuple[ParameterSpec, ...] = ()
    permission: Permission = Permission.EXECUTE
    handler: Callable[[dict[str, Any]], Any] = field(repr=False, kw_only=True)
    timeout_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializable schema (for listing / documentation)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "required": p.required,
                    "default": p.default,
                    "description": p.description,
                }
                for p in self.parameters
            ],
            "permission": self.permission.value,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class ToolCall:
    """An emitted tool invocation (name + validated arguments)."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    caller: str = ""
    call_id: str = ""
    timestamp: float = 0.0


@dataclass(frozen=True)
class ToolResult:
    """Structured outcome of dispatching a :class:`ToolCall`."""

    ok: bool
    name: str
    output: Any = None
    error: str = ""
    call_id: str = ""
    caller: str = ""
    run_id: str = ""
    timestamp: float = 0.0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tool": self.name,
            "output": self.output,
            "error": self.error,
            "call_id": self.call_id,
            "caller": self.caller,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
        }


__all__ = [
    "PARAM_TYPES",
    "ParameterSpec",
    "Permission",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    "permission_allows",
]
