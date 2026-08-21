"""tests/evaluation/test_geometry_metrics.py
===========================================
Unit tests for geometric accuracy metrics.
"""

from __future__ import annotations

import pytest

from cadgenesis.evaluation.geometry_metrics import GeometryMetrics
from sdk import toon_extended


def test_dimension_relative_error():
    predicted = {"w": 10.0, "h": 20.0}
    reference = {"w": 10.0, "h": 25.0}
    assert GeometryMetrics.dimension_relative_error(predicted, reference) == pytest.approx(0.1)
    assert GeometryMetrics.dimension_relative_error({"w": 1.0}, {"h": 2.0}) == 0.0
    assert GeometryMetrics.dimension_relative_error({}, {}) == 0.0
    assert GeometryMetrics.dimension_relative_error({"w": 1.0}, {"w": 0.0}) == pytest.approx(1.0)
    assert GeometryMetrics.dimension_relative_error(
        {"w": 1.0}, {"w": 1.0, "h": 3.0}
    ) == pytest.approx(0.0)


def test_bbox_iou():
    assert GeometryMetrics.bbox_iou((2, 2, 2), (2, 2, 2)) == 1.0
    assert GeometryMetrics.bbox_iou((2, 2, 2), (4, 4, 4)) == pytest.approx(0.125)
    assert GeometryMetrics.bbox_iou((2, 2, 2), (2, 2, 0)) == 0.0
    assert GeometryMetrics.bbox_iou((0, 0, 0), (1, 1, 1)) == 0.0
    assert GeometryMetrics.bbox_iou((-1, -1, -1), (1, 1, 1)) == 0.0


def test_validity_via_execution():
    valid_toon = toon_extended.to_toon(
        [{"primitive": "BOX", "w": "NUM_0", "h": "NUM_1", "d": "NUM_2"}]
    )
    fraction, error_counts = GeometryMetrics.validity_via_execution([valid_toon, "garbage", ""])
    assert fraction == pytest.approx(1 / 3)
    assert error_counts["parse_failed"] == 2
    fraction, error_counts = GeometryMetrics.validity_via_execution([])
    assert fraction == 0.0
    assert error_counts == {}


def test_symmetry_error():
    objects = [
        {"primitive": "BOX", "w": 10.0, "h": 10.0, "d": 10.0},
        {"primitive": "BOX", "w": 10.0, "h": 20.0, "d": 10.0},
    ]
    assert GeometryMetrics.symmetry_error(objects) == pytest.approx(0.5)
    assert GeometryMetrics.symmetry_error([]) == 0.0
    assert GeometryMetrics.symmetry_error([{"primitive": "SPHERE", "r": 5.0}]) == 0.0


def test_perpendicularity_error():
    objects = [
        {"primitive": "BOX", "angle": 90.0},
        {"primitive": "BOX", "angle": 45.0},
        {"primitive": "BOX"},
    ]
    assert GeometryMetrics.perpendicularity_error(objects) == pytest.approx(0.5)
    assert GeometryMetrics.perpendicularity_error([]) == 0.0
