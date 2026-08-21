"""Tests for the Pillar 8 feedback loop."""

from __future__ import annotations

from cadgenesis.execution import CADExecutionResult, FeedbackLoop
from cadgenesis.execution.geometry_validation import (
    GeometryCheck,
    GeometryValidationReport,
)


def test_dict_report_failure() -> None:
    items = FeedbackLoop().collect(
        {"geometry": {"valid": False, "failed": ["mesh:self_intersections"]}}
    )
    assert len(items) == 1
    assert items[0].severity == "error"
    assert "self_intersections" in items[0].message


def test_dict_report_pass_no_items() -> None:
    items = FeedbackLoop().collect({"geometry": {"valid": True, "failed": []}})
    assert items == []


def test_none_report_skipped() -> None:
    assert FeedbackLoop().collect({"geometry": None}) == []


def test_report_object_checks() -> None:
    report = GeometryValidationReport(
        checks=[
            GeometryCheck(
                "mesh:watertight",
                passed=False,
                severity="error",
                detail="open mesh",
                recommendation="fill holes",
            )
        ]
    )
    items = FeedbackLoop().collect({"geometry": report})
    assert items[0].suggestion == "fill holes"
    assert items[0].source == "geometry"


def test_summary_object_report() -> None:
    report = GeometryValidationReport(checks=[])
    items = FeedbackLoop().collect({"geometry": report})
    assert items == []


def test_apply_folds_into_result() -> None:
    result = CADExecutionResult()
    FeedbackLoop().apply(
        result,
        {"geometry": {"valid": False, "failed": ["mesh:self_intersections"]}},
    )
    assert any("self_intersections" in e for e in result.errors)


def test_apply_suggestion() -> None:
    result = CADExecutionResult()
    FeedbackLoop().apply(
        result,
        {
            "optimization": {
                "passed": True,
                "suggestions": ["reduce wall thickness"],
            }
        },
    )
    assert any("wall thickness" in s for s in result.suggestions)


def test_to_dict() -> None:
    data = FeedbackLoop().to_dict({"geometry": {"valid": False, "failed": ["mesh:watertight"]}})
    assert data[0]["source"] == "geometry"
    assert data[0]["severity"] == "error"
