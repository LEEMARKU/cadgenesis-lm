"""
tests/execution/test_failure_modes.py
=====================================
Tests for the formal failure-mode taxonomy
(pre-training gate: failure detection + diagnosis).
"""

from __future__ import annotations

from cadgenesis.execution.failure_modes import (
    FailureMode,
    classify_program,
    classify_reason,
    classify_result,
    failure_mode_counts,
)


class TestClassifyReason:
    def test_empty_token_list(self):
        assert classify_reason("empty token list") == FailureMode.MISSING_BASE_SOLID

    def test_missing_base_solid(self):
        assert classify_reason("missing base solid operation (EXTRUDE/BOX)") == FailureMode.MISSING_BASE_SOLID

    def test_geometry_invalid(self):
        assert classify_reason("geometry validation failed (analytic kernel)") == FailureMode.GEOMETRY_INVALID

    def test_non_manifold(self):
        assert classify_reason("non-manifold edges detected") == FailureMode.NON_MANIFOLD

    def test_underconstrained(self):
        assert classify_reason("system underconstrained: 3 degrees of freedom") == FailureMode.UNDERCONSTRAINED

    def test_infeasible(self):
        assert classify_reason("infeasible system: constraint conflict") == FailureMode.INFEASIBLE_CONSTRAINT

    def test_unknown_token(self):
        assert classify_reason("unknown token DOWEL") == FailureMode.UNKNOWN_TOKEN

    def test_timeout(self):
        assert classify_reason("tool timed out after 5s") == FailureMode.TIMEOUT

    def test_execution_error(self):
        assert classify_reason("validator error: bad op") == FailureMode.EXECUTION_ERROR

    def test_unclassified(self):
        assert classify_reason("something new and surprising") == FailureMode.UNKNOWN

    def test_empty_reason(self):
        assert classify_reason("") == FailureMode.UNKNOWN


class TestClassifyProgram:
    def test_empty(self):
        assert classify_program([]) == FailureMode.MISSING_BASE_SOLID

    def test_no_base_op(self):
        assert classify_program(["FILLET", "NUM_5"]) == FailureMode.MISSING_BASE_SOLID

    def test_extrude_without_dimension(self):
        assert classify_program(["SKETCH_RECT", "EXTRUDE", "BOX"]) == FailureMode.BAD_DIMENSION

    def test_extrude_with_dimension(self):
        assert classify_program(["SKETCH_RECT", "NUM_80", "EXTRUDE", "NUM_40", "BOX"]) == FailureMode.UNKNOWN

    def test_non_string_token(self):
        assert classify_program(["BOX", 5]) == FailureMode.EXECUTION_ERROR


class TestClassifyResult:
    def test_failure_reason_wins(self):
        result = type("R", (), {"failure_reason": "geometry validation failed"})()
        assert classify_result(result) == FailureMode.GEOMETRY_INVALID

    def test_errors_list(self):
        result = type("R", (), {"failure_reason": "", "errors": ["underconstrained system"]})()
        assert classify_result(result) == FailureMode.UNDERCONSTRAINED

    def test_repair_report_message(self):
        result = type(
            "R",
            (),
            {
                "failure_reason": "",
                "errors": [],
                "repair_report": {"message": "timeout after 5s"},
            },
        )()
        assert classify_result(result) == FailureMode.TIMEOUT

    def test_none_result(self):
        assert classify_result(None) == FailureMode.UNKNOWN


class TestFailureModeCounts:
    def test_counts(self):
        modes = [
            FailureMode.GEOMETRY_INVALID,
            FailureMode.GEOMETRY_INVALID,
            FailureMode.TIMEOUT,
            "unknown",
        ]
        counts = failure_mode_counts(modes)
        assert counts["geometry_invalid"] == 2
        assert counts["timeout"] == 1
        assert counts["unknown"] == 1