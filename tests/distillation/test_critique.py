"""tests/distillation/test_critique.py"""

from __future__ import annotations

from cadgenesis.distillation.critique import CritiqueEngine
from sdk import toon_extended

GOOD_TOON = toon_extended.to_toon(
    [{"id": 1, "feature": "BOX", "width": 50.0, "height": 30.0, "depth": 20.0, "fillet": 2.0}],
    include_schema=True,
)


def test_critique_passes_valid_toon():
    feedback = CritiqueEngine().critique(GOOD_TOON, "Design a mounting bracket.")
    assert feedback.score == 1.0
    assert feedback.issues == []
    assert feedback.suggestions == []


def test_critique_flags_negative_dimensions():
    bad = toon_extended.to_toon(
        [{"id": 1, "feature": "BOX", "width": -5.0, "height": 10.0, "depth": 10.0}],
        include_schema=True,
    )
    feedback = CritiqueEngine().critique(bad, "a box")
    assert feedback.score < 1.0
    assert any("width" in issue for issue in feedback.issues)


def test_critique_flags_unparsable_toon():
    feedback = CritiqueEngine().critique("not a toon at all", "a box")
    assert feedback.score == 0.0
    assert feedback.issues
    assert feedback.suggestions


def test_critique_flags_missing_expected_feature():
    cylinder_toon = toon_extended.to_toon(
        [{"id": 1, "feature": "CYLINDER", "width": 20.0, "height": 40.0, "depth": 20.0}],
        include_schema=True,
    )
    feedback = CritiqueEngine().critique(cylinder_toon, "Design a mounting bracket for a rod.")
    assert any("BOX" in issue for issue in feedback.issues)
    assert feedback.score < 1.0


def test_critique_accepts_matching_feature():
    cylinder_toon = toon_extended.to_toon(
        [{"id": 1, "feature": "CYLINDER", "width": 20.0, "height": 40.0, "depth": 20.0}],
        include_schema=True,
    )
    feedback = CritiqueEngine().critique(cylinder_toon, "Create a cylindrical housing.")
    assert feedback.score == 1.0


def test_critique_flags_oversized_fillet():
    bad = toon_extended.to_toon(
        [{"id": 1, "feature": "BOX", "width": 10.0, "height": 10.0, "depth": 10.0, "fillet": 9.0}],
        include_schema=True,
    )
    feedback = CritiqueEngine().critique(bad, "a box")
    assert any("fillet" in issue for issue in feedback.issues)
    assert feedback.score < 1.0


def test_critique_flags_empty_feature():
    bad = toon_extended.to_toon([{"id": 1, "feature": ""}], include_schema=True)
    feedback = CritiqueEngine().critique(bad, "a box")
    assert any("feature" in issue for issue in feedback.issues)


def test_build_feedback_from_errors_scoring():
    engine = CritiqueEngine(issue_penalty=0.25)
    feedback = engine.build_feedback_from_errors(GOOD_TOON, ["negative width", "empty feature"])
    assert feedback.score == 0.5
    assert len(feedback.suggestions) == 2
    assert feedback.issues == ["negative width", "empty feature"]
    heavy = engine.build_feedback_from_errors(GOOD_TOON, ["a", "b", "c", "d", "e"])
    assert heavy.score == 0.0


def test_score_clamped_to_zero():
    engine = CritiqueEngine()
    feedback = engine.build_feedback_from_errors(GOOD_TOON, ["x"] * 10)
    assert feedback.score == 0.0
