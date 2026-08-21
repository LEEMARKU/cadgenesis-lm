"""Tests for the M5 program-reasoning bridge (CAD-IR -> hybrid reasoning)."""

from __future__ import annotations

from cadgenesis.ir import parse_program
from cadgenesis.reasoning.constraint_solver import Constraint
from cadgenesis.reasoning.program_reasoning import (
    MAX_SIZE_MM,
    MIN_HOLE_MM,
    MIN_WALL_MM,
    ProgramReasoningEngine,
)

_BOX = ["BOX", "NUM_80", "NUM_40", "NUM_20", "EXTRUDE", "NUM_10"]
_CYLINDER = ["CYLINDER", "NUM_60", "NUM_30"]
_SPHERE = ["SPHERE", "NUM_50"]
_FEATURE_ONLY = ["EXTRUDE", "NUM_10"]
_TOO_THIN = ["BOX", "NUM_0"]


def _engine() -> ProgramReasoningEngine:
    return ProgramReasoningEngine()


def test_primitives_from_box_program() -> None:
    primitives = _engine().primitives(parse_program(_BOX))
    assert len(primitives) == 1
    box = primitives[0]
    assert box.kind == "box"
    assert box.dims == {"length": 80.0, "width": 40.0, "height": 20.0}
    assert box.name == "BOX"


def test_primitives_defaults_for_missing_params() -> None:
    primitives = _engine().primitives(parse_program(["BOX"]))
    assert len(primitives) == 1
    assert primitives[0].dims == {"length": 10.0, "width": 10.0, "height": 10.0}


def test_primitives_skip_feature_ops() -> None:
    assert _engine().primitives(parse_program(_FEATURE_ONLY)) == []


def test_cylinder_and_sphere_mapping() -> None:
    engine = _engine()
    cyl = engine.primitives(parse_program(_CYLINDER))[0]
    assert cyl.kind == "cylinder"
    assert cyl.dims == {"radius": 60.0, "height": 30.0}
    sphere = engine.primitives(parse_program(_SPHERE))[0]
    assert sphere.kind == "sphere"
    assert sphere.dims == {"radius": 50.0}


def test_context_key_is_program_id() -> None:
    program = parse_program(_BOX)
    context = _engine().context_for(program)
    assert context["id"] == program.program_id
    assert context["query"] == "BOX"
    assert context["min_dim"] == 10.0
    assert context["max_dim"] == 80.0


def test_reason_passes_valid_program() -> None:
    report = _engine().reason(parse_program(_BOX))
    assert report.passed
    assert report.score >= 0.5
    names = report.stage_names()
    assert "rules" in names and "constraints" in names and "manufacturing" in names
    geometry = report.stage("geometry")
    assert geometry is not None and geometry.passed


def test_reason_blocks_wall_too_thin() -> None:
    report = _engine().reason(parse_program(_TOO_THIN))
    assert not report.passed
    rules = report.stage("rules")
    assert rules is not None and not rules.passed
    assert "wall_too_thin" in rules.detail["errors"]
    manufacturing = report.stage("manufacturing")
    assert manufacturing is not None and not manufacturing.passed


def test_reason_blocks_oversize_program() -> None:
    engine = _engine()
    context = engine.context_for(parse_program(_BOX))
    context["max_dim"] = MAX_SIZE_MM + 100.0
    report = engine.pipeline.reason(context)
    assert not report.passed
    rules = report.stage("rules")
    assert rules is not None and "oversize_part" in rules.detail["errors"]


def test_hole_diameter_reported_from_hole_features() -> None:
    context = _engine().context_for(parse_program(["CYLINDER", "NUM_60", "HOLE", "NUM_2"]))
    assert context["hole_diameter"] == 2.0
    assert context["part"]["hole_diameter"] == 2.0


def test_hole_too_small_caught_by_rules_and_manufacturing() -> None:
    engine = _engine()
    context = engine.context_for(parse_program(["CYLINDER", "NUM_60", "HOLE", "NUM_5"]))
    context["hole_diameter"] = MIN_HOLE_MM - 0.5
    context["part"]["hole_diameter"] = MIN_HOLE_MM - 0.5
    report = engine.pipeline.reason(context)
    assert not report.passed
    rules = report.stage("rules")
    assert rules is not None and rules.passed
    assert "hole_too_small" in rules.detail["fired"]
    manufacturing = report.stage("manufacturing")
    assert manufacturing is not None and not manufacturing.passed


def test_repair_on_feasible_program_drops_nothing() -> None:
    result = _engine().repair(parse_program(_BOX))
    assert result["feasible"]
    assert result["dropped"] == []


def test_repair_relaxes_conflicting_extra_constraint() -> None:
    program = parse_program(_BOX)
    engine = _engine()
    box_width = Constraint("width_ge_500", {"BOX_0_1": 1.0}, ">=", 500.0)
    box_width_cap = Constraint("width_le_20", {"BOX_0_1": 1.0}, "<=", 20.0)
    result = engine.repair(program, extra_constraints=[box_width, box_width_cap])
    assert result["feasible"]
    assert len(result["dropped"]) == 1
    assert result["dropped"][0] in {"width_ge_500", "width_le_20"}
    width = result["assignment"]["BOX_0_1"]
    assert 0.8 <= width <= 1000.0


def test_benchmark_aggregates_stats() -> None:
    stats = _engine().benchmark([parse_program(_BOX), parse_program(_TOO_THIN)])
    assert stats["n"] == 2
    assert stats["passed"] == 1
    assert stats["pass_rate"] == 0.5
    assert stats["stage_failures"].get("rules", 0) == 1
    assert stats["mean_ms"] >= 0.0


def test_engine_thresholds_are_consistent_with_manufacturing_rules() -> None:
    engine = _engine()
    assert engine.manufacturing_rules.min_wall_thickness == MIN_WALL_MM
    assert engine.manufacturing_rules.min_hole_diameter == MIN_HOLE_MM
    assert engine.pipeline.threshold == 0.5
