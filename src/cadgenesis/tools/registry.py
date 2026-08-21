"""cadgenesis.tools.registry
===========================
Registry of tool definitions with schema and permission validation.
"""

from __future__ import annotations

from threading import Lock
from typing import Any

from cadgenesis.tools.schema import (
    ParameterSpec,
    Permission,
    ToolCall,
    ToolDefinition,
    permission_allows,
)


class ToolRegistry:
    """Name-indexed :class:`ToolDefinition` store with call validation.

    Thread-safe: registration and validation are guarded by a lock.
    """

    def __init__(self) -> None:
        self._defs: dict[str, ToolDefinition] = {}
        self._lock = Lock()

    def register(
        self,
        definition: ToolDefinition,
        replace: bool = False,
    ) -> ToolDefinition:
        """Register a tool definition (idempotent unless ``replace``)."""
        if not definition.name or not definition.name.isidentifier():
            raise ValueError(f"invalid tool name: {definition.name!r}")
        if not callable(definition.handler):
            raise TypeError(f"tool {definition.name!r}: handler must be callable")
        names = {p.name for p in definition.parameters}
        if len(names) != len(definition.parameters):
            raise ValueError(f"tool {definition.name!r}: duplicate parameter names")
        with self._lock:
            existing = self._defs.get(definition.name)
            if existing is not None and not replace:
                raise ValueError(f"tool {definition.name!r} already registered")
            self._defs[definition.name] = definition
        return definition

    def unregister(self, name: str) -> bool:
        """Remove a tool; returns True when it existed."""
        with self._lock:
            return self._defs.pop(name, None) is not None

    def get(self, name: str) -> ToolDefinition | None:
        """Look up a tool definition by name."""
        with self._lock:
            return self._defs.get(name)

    def names(self) -> list[str]:
        """Registered tool names, sorted."""
        with self._lock:
            return sorted(self._defs)

    def list_tools(self) -> list[dict[str, Any]]:
        """Serializable schemas of every registered tool, sorted by name."""
        with self._lock:
            return [self._defs[n].to_dict() for n in sorted(self._defs)]

    def validate_call(
        self,
        call: ToolCall,
        granted: Permission = Permission.EXECUTE,
    ) -> tuple[ToolDefinition, dict[str, Any]]:
        """Validate a call against the registry.

        Returns ``(definition, normalized_arguments)`` or raises
        ``ValueError`` (unknown tool / bad argument) or ``PermissionError``
        (caller permission below the tool's requirement).
        """
        with self._lock:
            definition = self._defs.get(call.name)
        if definition is None:
            raise ValueError(f"unknown tool: {call.name!r}")
        if not permission_allows(definition.permission, granted):
            raise PermissionError(
                f"tool {call.name!r} requires permission {definition.permission.value!r}, "
                f"caller granted {granted.value!r}"
            )
        normalized: dict[str, Any] = {}
        for spec in definition.parameters:
            if spec.name in call.arguments:
                value = call.arguments[spec.name]
                if not self._type_ok(spec, value):
                    raise ValueError(
                        f"tool {call.name!r}: parameter {spec.name!r} must be "
                        f"{spec.type}, got {type(value).__name__}"
                    )
                normalized[spec.name] = value
            elif spec.required:
                raise ValueError(f"tool {call.name!r}: missing required parameter {spec.name!r}")
            else:
                normalized[spec.name] = spec.default
        for key in call.arguments:
            if key not in {p.name for p in definition.parameters}:
                raise ValueError(f"tool {call.name!r}: unknown parameter {key!r}")
        return definition, normalized

    @staticmethod
    def _type_ok(spec: ParameterSpec, value: Any) -> bool:
        if spec.type == "string":
            return isinstance(value, str)
        if spec.type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if spec.type == "boolean":
            return isinstance(value, bool)
        if spec.type == "list":
            return isinstance(value, list)
        if spec.type == "program":
            return isinstance(value, list) and all(isinstance(t, str) for t in value)
        return False


__all__ = ["ToolRegistry"]
