"""tests/reasoning/test_symbolic_reasoner.py
============================================
Unit tests for cadgenesis.reasoning.symbolic_reasoner.
"""

from __future__ import annotations

import math

import pytest

from cadgenesis.reasoning.symbolic_reasoner import (
    SymbolicExpression,
    SymbolicReasoner,
    VerificationResult,
)


class TestSymbolicExpression:
    def test_basic_arithmetic(self):
        assert SymbolicExpression("2 + 3 * 4").evaluate() == 14.0

    def test_variables(self):
        expr = SymbolicExpression("w + 2 * t")
        assert expr.variables() == ["w", "t"]
        assert expr.evaluate({"w": 10.0, "t": 2.5}) == 15.0

    def test_constants(self):
        assert SymbolicExpression("pi").evaluate() == pytest.approx(math.pi)
        assert SymbolicExpression("2 * pi * r").evaluate({"r": 1.0}) == pytest.approx(2 * math.pi)

    def test_functions(self):
        assert SymbolicExpression("sqrt(16)").evaluate() == 4.0
        assert SymbolicExpression("min(3, 7)").evaluate() == 3.0

    def test_power(self):
        assert SymbolicExpression("2 ** 10").evaluate() == 1024.0

    def test_unknown_function_rejected(self):
        with pytest.raises(ValueError):
            SymbolicExpression("evil(__import__('os'))").evaluate()

    def test_unknown_variable_rejected(self):
        with pytest.raises(NameError):
            SymbolicExpression("nope").evaluate()

    def test_non_numeric_literal_rejected(self):
        with pytest.raises(TypeError):
            SymbolicExpression("'string'").evaluate()

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            SymbolicExpression("")

    def test_non_numeric_variable_rejected(self):
        with pytest.raises(TypeError):
            SymbolicExpression("x").evaluate({"x": "ten"})


class TestSymbolicReasoner:
    def test_evaluate(self):
        assert SymbolicReasoner.evaluate("d / 2", {"d": 6.0}) == 3.0

    def test_check_constraint_eq(self):
        result = SymbolicReasoner.check_constraint("w + 2*t", "==", 15.0, {"w": 10, "t": 2.5})
        assert isinstance(result, VerificationResult)
        assert result.passed

    def test_check_constraint_gt(self):
        result = SymbolicReasoner.check_constraint("thickness", ">=", 0.8, {"thickness": 1.2})
        assert result.passed
        assert not SymbolicReasoner.check_constraint(
            "thickness", ">=", 0.8, {"thickness": 0.5}
        ).passed

    def test_check_constraint_invalid_op(self):
        with pytest.raises(ValueError):
            SymbolicReasoner.check_constraint("x", "≈", 0.0)

    def test_check_implication(self):
        result = SymbolicReasoner.check_implication("w + 2*t", "d", {"w": 4, "t": 1, "d": 6})
        assert result.passed

    def test_check_token_consistency(self):
        def decode(token):
            if token.startswith("NUM_"):
                return float(token[4:])
            return None

        ok, _ = SymbolicReasoner.check_token_consistency(["NUM_25", "NUM_10"], decode)
        assert ok

        bad, msg = SymbolicReasoner.check_token_consistency(["NUM_-5"], decode)
        assert not bad
        assert "negative" in msg
