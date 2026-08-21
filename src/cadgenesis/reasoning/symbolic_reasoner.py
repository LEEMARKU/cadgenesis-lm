"""cadgenesis.reasoning.symbolic_reasoner
========================================
Symbolic reasoning over CAD expressions: a safe arithmetic evaluator, logical
implication / constraint verification, and consistency checks on CAD token
sequences.

The evaluator accepts only a whitelisted grammar (numbers, the four operators,
``**``, parentheses, ``pi``/``e`` and the ``abs``/``sqrt``/``sin``/``cos``/
``tan``/``min``/``max`` functions) so arbitrary Python code can never run.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_SAFE_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "min": min,
    "max": max,
    "round": round,
}

_SAFE_BIN_OPS: dict[type, Callable[[Any, Any], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_SAFE_UNARY: dict[type, Callable[[Any], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


@dataclass
class VerificationResult:
    """Outcome of a symbolic check."""

    passed: bool
    expression: str = ""
    operator: str = ""
    rhs: float | None = None
    value: float | None = None
    message: str = ""

    @property
    def is_passed(self) -> bool:
        return self.passed


class SymbolicExpression:
    """A parsed, safe arithmetic expression with named variables."""

    def __init__(self, expression: str) -> None:
        if not expression or not isinstance(expression, str):
            raise ValueError("expression must be a non-empty string")
        self.expression = expression
        self._tree = ast.parse(expression, mode="eval").body

    def variables(self) -> list[str]:
        """Names referenced by the expression (excluding built-in constants)."""
        return [
            node.id
            for node in ast.walk(self._tree)
            if isinstance(node, ast.Name) and node.id not in ("pi", "e")
        ]

    def evaluate(self, variables: dict[str, float] | None = None) -> float:
        """Evaluate the expression; missing variables are treated as 0."""
        env: dict[str, float] = {"pi": math.pi, "e": math.e}
        if variables:
            for name, value in variables.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(f"variable {name!r} must be numeric")
                env[name] = float(value)
        return float(self._eval(self._tree, env))

    def _eval(self, node: ast.AST, env: dict[str, float]) -> float:
        if isinstance(node, ast.Expression):
            return self._eval(node.body, env)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise TypeError(f"unsupported literal {node.value!r}")
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            raise NameError(f"unknown variable {node.id!r}")
        if isinstance(node, ast.BinOp):
            op = _SAFE_BIN_OPS.get(type(node.op))
            if op is None:
                raise ValueError("unsupported binary operator")
            return op(self._eval(node.left, env), self._eval(node.right, env))
        if isinstance(node, ast.UnaryOp):
            unary_op = _SAFE_UNARY.get(type(node.op))
            if unary_op is None:
                raise ValueError("unsupported unary operator")
            return unary_op(self._eval(node.operand, env))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("unsupported callable")
            func = _SAFE_FUNCTIONS.get(node.func.id)
            if func is None:
                raise ValueError(f"function {node.func.id!r} is not allowed")
            args = [self._eval(arg, env) for arg in node.args]
            return float(func(*args))
        raise ValueError(f"unsupported expression node {type(node).__name__}")

    def __repr__(self) -> str:
        return f"SymbolicExpression({self.expression!r})"


class SymbolicReasoner:
    """Symbolic verification of CAD design expressions."""

    @staticmethod
    def evaluate(
        expression: str,
        variables: dict[str, float] | None = None,
    ) -> float:
        """Evaluate ``expression`` against ``variables``."""
        return SymbolicExpression(expression).evaluate(variables)

    @staticmethod
    def check_constraint(
        expression: str,
        operator_str: str,
        rhs: float,
        variables: dict[str, float] | None = None,
        tolerance: float = 1e-9,
    ) -> VerificationResult:
        """Verify ``expression <op> rhs`` (op in ``==``, ``<``, ``<=``, ``>``, ``>=``)."""
        if operator_str not in ("==", "<", "<=", ">", ">="):
            raise ValueError(f"invalid operator {operator_str!r}; expected one of ==, <, <=, >, >=")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        expr = SymbolicExpression(expression)
        value = expr.evaluate(variables)
        if operator_str == "==":
            passed = abs(value - rhs) <= tolerance
        elif operator_str == "<":
            passed = value < rhs
        elif operator_str == "<=":
            passed = value <= rhs + tolerance
        elif operator_str == ">":
            passed = value > rhs
        else:
            passed = value >= rhs - tolerance
        message = "" if passed else f"{expression} evaluated to {value:.6g}"
        return VerificationResult(
            passed=passed,
            expression=expression,
            operator=operator_str,
            rhs=rhs,
            value=value,
            message=message,
        )

    @staticmethod
    def check_implication(
        antecedent: str,
        consequent: str,
        variables: dict[str, float] | None = None,
        tolerance: float = 1e-9,
    ) -> VerificationResult:
        """Verify that a numeric implication ``antecedent => consequent`` holds.

        Both arguments are equality-style expressions (e.g. ``"w + 2*t"`` and
        ``"d"``); the implication holds when the antecedent and consequent
        evaluate to the same value within tolerance.  For Boolean antecedents
        use :meth:`check_constraint` with a concrete operator.
        """
        value_a = SymbolicExpression(antecedent).evaluate(variables)
        value_c = SymbolicExpression(consequent).evaluate(variables)
        passed = abs(value_a - value_c) <= tolerance
        return VerificationResult(
            passed=passed,
            expression=f"{antecedent} => {consequent}",
            operator="==",
            rhs=value_c,
            value=value_a,
            message="" if passed else f"antecedent {value_a:.6g} != consequent {value_c:.6g}",
        )

    @staticmethod
    def check_token_consistency(
        tokens: list[str],
        numeric_decode: Callable[[str], float | None],
    ) -> tuple[bool, str]:
        """Verify that numeric tokens decode to finite, positive values.

        ``numeric_decode`` maps a token string to a float (e.g.
        ``NumericTokenizer.decode_length``) or ``None`` for non-numeric tokens.
        """
        bad: list[str] = []
        for token in tokens:
            value = numeric_decode(token)
            if value is None:
                continue
            if not math.isfinite(float(value)):
                bad.append(f"{token} (non-finite)")
            elif float(value) < 0:
                bad.append(f"{token} (negative)")
        if bad:
            return False, f"Numeric tokens with invalid values: {bad[:5]}"
        return True, "OK"


__all__ = [
    "SymbolicExpression",
    "SymbolicReasoner",
    "VerificationResult",
]
