"""tests/evaluation/test_cad_metrics.py
=======================================
Unit tests for CAD-generation metrics (validity, fidelity, coverage).
"""

from __future__ import annotations

import pytest

from cadgenesis.evaluation.cad_metrics import CADMetrics
from sdk import toon_extended


def _box_toon() -> str:
    return toon_extended.to_toon([{"primitive": "BOX", "w": "NUM_0", "h": "NUM_1", "d": "NUM_2"}])


def test_validity_rate_valid():
    assert CADMetrics.validity_rate([_box_toon(), _box_toon()]) == 1.0


def test_validity_rate_garbage():
    assert CADMetrics.validity_rate(["not a toon at all", ""]) == 0.0


def test_validity_rate_empty():
    assert CADMetrics.validity_rate([]) == 0.0


def test_token_accuracy():
    predicted = [["BOX", "NUM_1"], ["SPHERE", "NUM_2"]]
    reference = [["BOX", "NUM_1"], ["SPHERE", "NUM_3"]]
    assert CADMetrics.token_accuracy(predicted, reference) == pytest.approx(0.75)
    assert CADMetrics.token_accuracy([], []) == 0.0
    assert CADMetrics.token_accuracy([["BOX"]], [[]]) == 0.0


def test_edit_distance_similarity():
    assert CADMetrics.edit_distance_similarity(["BOX", "NUM_1"], ["BOX", "NUM_1"]) == 1.0
    assert CADMetrics.edit_distance_similarity(["abcd"], ["efgh"]) == 0.0
    assert CADMetrics.edit_distance_similarity([], []) == 1.0
    assert CADMetrics.edit_distance_similarity(["abc"], ["abcd"]) == pytest.approx(0.75)
    assert CADMetrics.edit_distance_similarity(["kitten"], ["sitting"]) == pytest.approx(
        1.0 - 3 / 7
    )


def test_primitive_coverage():
    toons = [
        toon_extended.to_toon([{"primitive": "BOX", "w": "NUM_0"}]),
        toon_extended.to_toon([{"primitive": "CYLINDER", "r": "NUM_1"}]),
        toon_extended.to_toon(
            [
                {"primitive": "SKETCH_RECT", "w": "NUM_2", "h": "NUM_3"},
                {"primitive": "EXTRUDE", "d": "NUM_4"},
            ]
        ),
        "garbage",
    ]
    coverage = CADMetrics.primitive_coverage(toons)
    assert coverage["BOX"] == pytest.approx(0.25)
    assert coverage["CYLINDER"] == pytest.approx(0.25)
    assert coverage["SPHERE"] == 0.0
    assert coverage["EXTRUDE_PROFILE"] == pytest.approx(0.25)
    assert CADMetrics.primitive_coverage([]) == {
        "BOX": 0.0,
        "CYLINDER": 0.0,
        "SPHERE": 0.0,
        "EXTRUDE_PROFILE": 0.0,
    }


def test_plan_accuracy():
    executed = [["BOX", "EXTRUDE"], ["BOX"]]
    planned = [["BOX", "EXTRUDE"], ["BOX", "SPHERE"]]
    assert CADMetrics.plan_accuracy(executed, planned) == pytest.approx(0.75)
    assert CADMetrics.plan_accuracy([], []) == 0.0
    assert CADMetrics.plan_accuracy([["BOX"]], [[]]) == 0.0
