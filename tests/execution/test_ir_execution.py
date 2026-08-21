"""
tests/execution/test_ir_execution.py
====================================
v6.4 IR-native execution: typed program graphs materialise into the world
model, expose a queryable state, and feed an IR-diff feedback loop — with
no token-stream round-trip.
"""

from __future__ import annotations

import pytest

from cadgenesis.execution import (
    CADExecutionEngine,
    FeedbackItem,
    IRExecutionEngine,
)
from cadgenesis.ir import parse_program, toon_to_program
from cadgenesis.ir.diff import IrDiffReport, ir_diff
from cadgenesis.tokenizer import AutonomousCADTokenizer


def box_fillet_program():
    return toon_to_program(
        "id|feature|width|height|depth|fillet\n"
        "int|str|float|float|float|float\n"
        "1|BOX|50.0|30.0|20.0|2.0"
    )


class TestIRExecutionEngine:
    def test_materialises_box_with_world_model_math(self):
        result = IRExecutionEngine().execute(box_fillet_program())
        assert result.valid
        state = result.state
        assert len(state.objects) == 1
        obj = state.objects[0]
        assert obj.feature == "block"
        assert obj.params["length"] == 50.0
        assert obj.params["width"] == 30.0
        assert obj.params["height"] == 20.0
        assert obj.volume_m3 == pytest.approx(50 * 30 * 20 * 1e-9)
        assert obj.mass_kg == 0.0  # no material -> honest zero

    def test_cylinder_radius_is_half_width(self):
        program = toon_to_program(
            "id|feature|width|height\nint|str|float|float\n1|CYLINDER|10.0|40.0"
        )
        state = IRExecutionEngine().execute(program).state
        assert state.objects[0].feature == "cylinder"
        assert state.objects[0].params["radius"] == 5.0
        assert state.objects[0].params["height"] == 40.0

    def test_sphere_radius_is_half_width(self):
        program = toon_to_program(
            "id|feature|width\nint|str|float\n1|SPHERE|20.0"
        )
        state = IRExecutionEngine().execute(program).state
        assert state.objects[0].feature == "sphere"
        assert state.objects[0].params["radius"] == 10.0

    def test_material_yields_mass(self):
        from cadgenesis.world_model import Material

        program = toon_to_program(
            "id|feature|width|height|depth\nint|str|float|float|float\n1|BOX|100.0|100.0|100.0"
        )
        state = IRExecutionEngine(material=Material()).execute(program).state
        obj = state.objects[0]
        assert obj.mass_kg == pytest.approx(obj.volume_m3 * 7850.0)

    def test_fillet_feature_accumulates_on_parent(self):
        from cadgenesis.ir import decode_param_value

        program = parse_program(["BOX", "NUM_050", "FEAT_FILLET", "NUM_005"])
        state = IRExecutionEngine().execute(program).state
        obj = state.objects[0]
        assert obj.kind == "PRIM_BOX"
        # NUM_005 is a zero-padded quantizer bin, not 5.0 mm
        assert obj.params["fillet"] == pytest.approx(decode_param_value("NUM_005"))
        assert obj.applied_features[-1]["kind"] == "FEAT_FILLET"

    def test_orphan_feature_is_unresolved_not_crashed(self):
        from cadgenesis.ir import CadOperation, CadProgram, operation_id

        # A manually-built program can pass the structural gate (the RAW
        # step carries the "BOX" base token) while the FEAT step's dependency
        # never materialises: both ops land in `unresolved`.
        raw = CadOperation(
            op_id=operation_id("RAW", {}, 0),
            kind="RAW",
            params={},
            tokens=("BOX",),
            position=0,
        )
        feat = CadOperation(
            op_id=operation_id("FEAT_FILLET", {"d0": 5.0}, 1),
            kind="FEAT_FILLET",
            params={"d0": 5.0},
            depends_on=(raw.op_id,),
            tokens=("FEAT_FILLET",),
            position=1,
        )
        result = IRExecutionEngine().execute(CadProgram.build([raw, feat]))
        assert result.valid
        assert result.state.objects == []
        assert len(result.state.unresolved) == 2

    def test_unmappable_kind_is_unresolved(self):
        program = parse_program(["BOX", "NUM_050", "FEAT_HOLE", "NUM_010"])
        result = IRExecutionEngine().execute(program)
        state = result.state
        assert state.objects[0].kind == "PRIM_BOX"
        assert state.objects[0].applied_features[-1]["kind"] == "FEAT_HOLE"

    def test_dependency_order_is_respected(self):
        from cadgenesis.ir import decode_param_value

        # FEAT_FILLET depends_on the box op: applied to it even though a
        # second primitive follows.
        program = parse_program(["BOX", "NUM_050", "FEAT_FILLET", "NUM_005", "CYLINDER", "NUM_020"])
        state = IRExecutionEngine().execute(program).state
        box, cyl = state.objects
        assert box.params["fillet"] == pytest.approx(decode_param_value("NUM_005"))
        assert "fillet" not in cyl.params

    def test_invalid_program_is_rejected(self):
        from cadgenesis.ir import CadOperation, CadProgram

        op = CadOperation(op_id="x", kind="PRIM_BOX", params={}, depends_on=("ghost",))
        result = IRExecutionEngine().execute(CadProgram.build([op]))
        assert not result.valid
        assert any("dependencies_resolve" in e for e in result.errors)
        assert result.state is None

    def test_vocab_gate_rejects_unregistered_tokens(self):
        program = parse_program(["BOX", "NUM_050", "FEAT_FILLET", "NUM_005"])
        result = IRExecutionEngine(vocab=AutonomousCADTokenizer.build_mini().vocab).execute(program)
        assert not result.valid
        assert any("tokens_registered" in e for e in result.errors)

    def test_query_api(self):
        state = IRExecutionEngine().execute(box_fillet_program()).state
        assert state.object(state.objects[0].op_id) is state.objects[0]
        assert state.object("nope") is None
        assert state.objects_of("PRIM_BOX") == [state.objects[0]]
        assert state.bounds() is not None
        assert state.total_volume() == pytest.approx(3e-5)

    def test_empty_program_rejected_by_gate(self):
        from cadgenesis.ir import CadProgram

        result = IRExecutionEngine().execute(CadProgram.build([]))
        assert not result.valid
        assert any("steps_non_empty" in e for e in result.errors)

    def test_requires_cadprogram_type(self):
        with pytest.raises(TypeError):
            IRExecutionEngine().execute(["BOX", "NUM_050"])  # type: ignore[arg-type]


class TestIrDiff:
    GOOD = (
        "id|feature|width|height|depth|fillet\n"
        "int|str|float|float|float|float\n"
        "1|BOX|50.0|30.0|20.0|2.0"
    )

    def test_identical_programs_no_changes(self):
        a = toon_to_program(self.GOOD)
        rebuilt = toon_to_program(self.GOOD)
        assert rebuilt.program_id == a.program_id  # content-hash determinism
        report = ir_diff(a, rebuilt)
        assert not report.has_changes
        assert report.unchanged == 1

    def test_added_removed_changed(self):
        before = toon_to_program(
            "id|feature|width|height\nint|str|float|float\n1|BOX|50.0|30.0"
        )
        after = toon_to_program(
            "id|feature|width|height\nint|str|float|float\n1|BOX|60.0|30.0\n2|SPHERE|10.0"
        )
        report = ir_diff(before, after)
        assert report.has_changes
        assert [o["kind"] for o in report.added] == ["PRIM_SPHERE"]
        assert report.removed == []
        assert len(report.changed) == 1
        assert report.changed[0]["changed_params"] == ["d0"]
        assert report.changed[0]["before_params"]["d0"] == 50.0
        assert report.changed[0]["after_params"]["d0"] == 60.0

    def test_removed_op(self):
        before = toon_to_program(
            "id|feature|width\nint|str|float\n1|BOX|50.0\n2|SPHERE|10.0"
        )
        after = toon_to_program("id|feature|width\nint|str|float\n1|BOX|50.0")
        report = ir_diff(before, after)
        assert [o["kind"] for o in report.removed] == ["PRIM_SPHERE"]
        assert report.unchanged == 1

    def test_summary_and_to_dict(self):
        report = ir_diff(box_fillet_program(), box_fillet_program())
        summary = report.summary()
        assert summary["has_changes"] is False
        assert report.to_dict()["added_ops"] == []


class TestEngineIntegration:
    def test_execute_ir_fills_result(self):
        engine = CADExecutionEngine()
        result = engine.execute_ir(box_fillet_program())
        assert result.is_valid_geometry
        assert result.confidence_score >= 0.5
        assert result.ir_report["valid"] is True
        assert result.ir_report["objects"] == 1
        assert result.ir_report["total_volume_m3"] == pytest.approx(3e-5)
        assert result.cost_breakdown["material_usd"] > 0.0

    def test_execute_ir_rejects_invalid(self):
        from cadgenesis.ir import CadOperation, CadProgram

        op = CadOperation(op_id="x", kind="PRIM_BOX", params={}, depends_on=("ghost",))
        result = CADExecutionEngine().execute_ir(CadProgram.build([op]))
        assert not result.is_valid_geometry
        assert result.confidence_score == 0.0
        assert result.ir_report["valid"] is False
        assert result.errors

    def test_diff_feedback_loop(self):
        before = toon_to_program(
            "id|feature|width|height\nint|str|float|float\n1|BOX|50.0|30.0"
        )
        after = toon_to_program(
            "id|feature|width|height\nint|str|float|float\n1|BOX|50.0|30.0\n2|SPHERE|10.0"
        )
        result = CADExecutionEngine().execute_ir(after, previous=before)
        assert any("added PRIM_SPHERE" in s for s in result.suggestions)
        assert result.ir_report["diff"]["added"]  # op dict present

    def test_warning_severity_for_removed(self):
        from cadgenesis.execution import FeedbackLoop

        report = IrDiffReport(
            before_id="a",
            after_id="b",
            removed=[{"kind": "PRIM_BOX", "position": 0}],
        )
        items = FeedbackLoop().feedback_on_diff(report)
        assert len(items) == 1
        assert items[0].severity == "warning"
        assert isinstance(items[0], FeedbackItem)

    def test_vocab_gate_through_engine(self):
        program = parse_program(["BOX", "NUM_050", "FEAT_FILLET", "NUM_005"])
        result = CADExecutionEngine().execute_ir(
            program, vocab=AutonomousCADTokenizer.build_mini().vocab
        )
        assert not result.is_valid_geometry
        assert any("[ir]" in e for e in result.errors)

    def test_type_errors(self):
        engine = CADExecutionEngine()
        with pytest.raises(TypeError):
            engine.execute_ir(["BOX", "NUM_050"])  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            engine.execute_ir(box_fillet_program(), previous=["BOX"])  # type: ignore[arg-type]

    def test_ir_result_contract(self):
        from cadgenesis.execution import CADExecutionResult

        result = CADExecutionResult()
        assert result.ir_report == {}
        assert "ir_report" in result.to_dict()