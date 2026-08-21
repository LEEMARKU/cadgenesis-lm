"""cadgenesis.cad.parametric.parameters
====================================
Parametric design variables and expression resolution.

A :class:`Parameter` is a named, unit-annotated value that may be referenced
by feature parameters throughout a part.  The :class:`ParameterTable` tracks
dependencies and resolves values (numeric or simple expressions such as
``"w * 2"`` or ``"box_depth + 5"``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

_VALID_UNITS = ("", "mm", "cm", "m", "deg", "rad", "g", "kg", "s", "%")

_EXPR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class Parameter:
    """A named parametric variable with units and optional expression."""

    name: str
    value: float = 0.0
    units: str = "mm"
    expression: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.name):
            raise ValueError(f"invalid parameter name {self.name!r}")
        if self.units not in _VALID_UNITS:
            raise ValueError(f"invalid units {self.units!r}")
        if self.expression is not None:
            self.expression = str(self.expression)
            if not self.expression.strip():
                self.expression = None
        if self.expression is None and not math.isfinite(float(self.value)):
            raise ValueError(f"parameter {self.name!r} must have a finite value")

    @property
    def is_expression(self) -> bool:
        return self.expression is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "units": self.units,
            "expression": self.expression,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Parameter:
        return cls(
            name=str(data["name"]),
            value=float(data.get("value", 0.0)),
            units=str(data.get("units", "mm")),
            expression=data.get("expression"),
            description=str(data.get("description", "")),
        )


class ParameterTable:
    """Ordered collection of :class:`Parameter` with dependency resolution."""

    def __init__(self, parameters: list[Parameter] | None = None) -> None:
        self._parameters: dict[str, Parameter] = {}
        for parameter in parameters or []:
            self.add(parameter)

    # -- mutation -----------------------------------------------------------
    def add(self, parameter: Parameter) -> Parameter:
        if parameter.name in self._parameters:
            raise KeyError(f"parameter {parameter.name!r} already exists")
        self._parameters[parameter.name] = parameter
        return parameter

    def set(
        self,
        name: str,
        value: float | None = None,
        expression: str | None = None,
    ) -> Parameter:
        parameter = self._parameters.get(name)
        if parameter is None:
            parameter = Parameter(name, value if value is not None else 0.0)
            self.add(parameter)
        if value is not None:
            parameter.value = float(value)
            parameter.expression = None
        if expression is not None:
            parameter.expression = str(expression)
        return parameter

    def remove(self, name: str) -> Parameter:
        if name not in self._parameters:
            raise KeyError(f"parameter {name!r} not found")
        return self._parameters.pop(name)

    # -- access -------------------------------------------------------------
    def __contains__(self, name: object) -> bool:
        return name in self._parameters

    def __getitem__(self, name: str) -> Parameter:
        return self._parameters[name]

    def get(self, name: str) -> Parameter | None:
        return self._parameters.get(name)

    def names(self) -> list[str]:
        return list(self._parameters)

    def values(self) -> list[Parameter]:
        return list(self._parameters.values())

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.values())

    # -- resolution ----------------------------------------------------------
    def dependencies(self, name: str) -> list[str]:
        """Parameter names referenced by ``name``'s expression."""
        parameter = self._parameters.get(name)
        if parameter is None or parameter.expression is None:
            return []
        return [
            m
            for m in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", parameter.expression)
            if m in self._parameters and m != name
        ]

    def resolve(self, name: str, _stack: tuple[str, ...] = ()) -> float:
        """Evaluate a parameter to a float, following expressions recursively."""
        parameter = self._parameters.get(name)
        if parameter is None:
            raise KeyError(f"parameter {name!r} not found")
        if name in _stack:
            raise ValueError(f"circular expression dependency: {(*_stack, name)}")
        if parameter.expression is None:
            return float(parameter.value)
        namespace: dict[str, float] = {}
        for dependency in self.dependencies(name):
            namespace[dependency] = self.resolve(dependency, (*_stack, name))
        safe = parameter.expression
        for key, value in namespace.items():
            safe = re.sub(rf"\b{key}\b", repr(value), safe)
        try:
            return float(eval(safe, {"__builtins__": {}}, namespace))
        except (SyntaxError, ZeroDivisionError, TypeError, ValueError) as exc:
            raise ValueError(
                f"cannot evaluate expression {parameter.expression!r} for {name!r}: {exc}"
            ) from exc

    def resolve_all(self) -> dict[str, float]:
        return {name: self.resolve(name) for name in self._parameters}

    def to_dict(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self.values()]

    @classmethod
    def from_dict(cls, data: list[dict[str, Any]]) -> ParameterTable:
        return cls([Parameter.from_dict(item) for item in data])


__all__ = ["Parameter", "ParameterTable"]
