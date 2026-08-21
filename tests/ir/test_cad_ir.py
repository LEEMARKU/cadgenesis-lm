"""
tests/ir/test_cad_ir.py
=======================
M2 CAD-IR: typed schema, lossless parsing, validation, serialization,
versioning, and unique content IDs.
"""

from __future__ import annotations

import pytest

from cadgenesis.datasets.cad_program_synth import build_synthetic_records
from cadgenesis.ir import (
    CAD_IR_SCHEMA_VERSION,
    CadOperation,
    CadProgram,
    decode_param_value,
    is_base_token,
    is_feature_token,
    is_schema_compatible,
    parse_program,
    validate_cad_program,
    validate_program_ir,
)

REPRESENTATIVE = [
    ["SKETCH_RECT", "NUM_80", "EXTRUDE", "NUM_10", "BOX"],
    ["SKETCH_RECT", "NUM_80", "EXTRUDE", "NUM_10", "BOX", "CYLINDER", "NUM_5"],
    ["CYLINDER", "NUM_5", "EXTRUDE", "NUM_10", "BOX"],
    ["CYLINDER", "NUM_10", "COUNTERBORE", "NUM_10"],
    ["PART", "NUM_10", "NUM_5", "EXTRUDE", "NUM_10"],
    ["THREAD", "NUM_10", "NUM_5", "CYLINDER"],
    ["BRACKET", "NUM_10", "NUM_5", "NUM_4", "MOUNT", "EXTRUDE", "SLOT"],
]

#: No base keyword at all -> rejected by both the legacy and IR gates.
NO_BASE = ["STEEL", "NUM_10", "NUM_5", "NUM_4", "WEIGHT", "VOLUME"]


class TestSchema:
    def test_base_keyword_classification(self):
        for token in ("BOX", "CYLINDER", "SPHERE", "SKETCH_RECT", "SKETCH", "PRIM_BOX"):
            assert is_base_token(token), token

    def test_feature_keyword_classification(self):
        for token in ("EXTRUDE", "HOLE", "THREAD", "PATTERN", "FILLET", "SLOT", "FEAT_HOLE"):
            assert is_feature_token(token), token

    def test_base_and_feature_are_disjoint(self):
        from cadgenesis.ir import BASE_KEYWORDS, FEATURE_KEYWORDS

        assert not (BASE_KEYWORDS & FEATURE_KEYWORDS)

    def test_decode_raw_mm_token(self):
        assert decode_param_value("NUM_80") == 80.0

    def test_decode_bin_token(self):
        value = decode_param_value("NUM_039")
        assert value is not None and 150.0 <= value <= 160.0

    def test_decode_angle_and_ratio(self):
        angle = decode_param_value("ANG_090")
        ratio = decode_param_value("RAT_000")
        assert angle is not None and 89.5 <= angle <= 91.5
        assert ratio is not None and 0.0 <= ratio <= 0.1

    def test_decode_unknown_token_is_none(self):
        assert decode_param_value("BOX") is None
        assert decode_param_value("bogus") is None


class TestParse:
    @pytest.mark.parametrize("tokens", REPRESENTATIVE)
    def test_round_trip_is_lossless(self, tokens):
        program = parse_program(tokens)
        assert program.to_tokens() == tokens

    def test_base_feature_chain(self):
        program = parse_program(["SKETCH_RECT", "NUM_80", "EXTRUDE", "NUM_10", "BOX"])
        kinds = [s.kind for s in program.steps]
        assert kinds == ["PRIM_BOX", "FEAT_EXTRUDE", "PRIM_BOX"]

    def test_dependency_chain(self):
        program = parse_program(["SKETCH_RECT", "NUM_80", "EXTRUDE", "NUM_10", "BOX"])
        assert len(program.steps) == 3
        assert program.steps[0].depends_on == ()
        assert program.steps[1].depends_on == (program.steps[0].op_id,)
        assert program.steps[2].depends_on == (program.steps[1].op_id,)

    def test_numeric_params_decoded(self):
        program = parse_program(["SKETCH_RECT", "NUM_80", "EXTRUDE", "NUM_10", "BOX"])
        assert program.steps[0].params["d0"] == 80.0
        assert program.steps[1].params["d0"] == 10.0

    def test_attributes_captured(self):
        program = parse_program(["STEEL", "NUM_10", "WEIGHT", "VOLUME"])
        raw = program.steps[0]
        assert raw.kind == "RAW"
        assert raw.params["attr"] == ["STEEL", "WEIGHT", "VOLUME"]

    def test_raw_op_when_no_base_opens(self):
        program = parse_program(["PART", "NUM_10", "NUM_5"])
        assert program.steps[0].kind == "RAW"

    def test_empty_program_has_no_steps(self):
        program = parse_program([])
        assert program.steps == ()
        assert program.to_tokens() == []

    def test_unknown_token_round_trips(self):
        tokens = ["BOX", "QUANTUM_WIDGET"]
        assert parse_program(tokens).to_tokens() == tokens

    def test_topological_order_is_complete(self):
        program = parse_program(["SKETCH_RECT", "NUM_80", "EXTRUDE", "NUM_10", "BOX"])
        order = program.topological_order()
        assert set(order) == {s.op_id for s in program.steps}
        assert order[0] == program.steps[0].op_id

    def test_is_cyclic_false_for_chain(self):
        assert not parse_program(["BOX", "EXTRUDE", "NUM_10"]).is_cyclic()


class TestValidation:
    @pytest.mark.parametrize("tokens", REPRESENTATIVE)
    def test_representative_programs_pass(self, tokens):
        assert validate_cad_program(tokens), tokens

    def test_missing_base_is_rejected(self):
        assert not validate_cad_program(["HOLE", "NUM_10"])

    def test_no_base_program_is_rejected(self):
        from cadgenesis.execution.geometry_validation import validate_program

        assert not validate_program(NO_BASE)
        assert not validate_cad_program(NO_BASE)

    def test_unknown_dependency_is_rejected(self):
        program = parse_program(["BOX"])
        steps = list(program.steps)
        steps.append(
            CadOperation(
                op_id="cadop:ghost",
                kind="FEAT_HOLE",
                params={},
                depends_on=("cadop:nope",),
                tokens=("HOLE",),
                position=1,
            )
        )
        report = validate_program_ir(CadProgram.build(steps))
        assert not report.passed
        assert any(c.name == "dependencies_resolve" and not c.passed for c in report.checks)

    def test_duplicate_op_ids_are_rejected(self):
        steps = [
            CadOperation(op_id="cadop:same", kind="PRIM_BOX", tokens=("BOX",), position=0),
            CadOperation(op_id="cadop:same", kind="FEAT_HOLE", tokens=("HOLE",), position=1),
        ]
        report = validate_program_ir(CadProgram.build(steps))
        assert not report.passed
        assert any(c.name == "op_ids_unique" and not c.passed for c in report.checks)

    def test_out_of_range_params_are_rejected(self):
        steps = [
            CadOperation(
                op_id="cadop:x",
                kind="PRIM_BOX",
                params={"d0": 5000.0},
                tokens=("BOX",),
                position=0,
            )
        ]
        report = validate_program_ir(CadProgram.build(steps))
        assert not report.passed
        assert any(c.name == "params_in_range" and not c.passed for c in report.checks)

    def test_round_trip_violation_is_reported(self):
        program = parse_program(["BOX"])
        report = validate_program_ir(program, original=["BOX", "EXTRA"])
        assert not report.passed
        assert any(c.name == "round_trip" and not c.passed for c in report.checks)

    def test_report_summary(self):
        report = validate_program_ir(parse_program(["BOX"]))
        assert report.passed
        assert "CAD-IR OK" in report.summary()


class TestSerialization:
    def test_to_dict_from_dict_round_trip(self):
        program = parse_program(["SKETCH_RECT", "NUM_80", "EXTRUDE", "NUM_10", "BOX"])
        restored = CadProgram.from_dict(program.to_dict())
        assert restored == program
        assert restored.program_id == program.program_id

    def test_to_json_from_json_round_trip(self):
        program = parse_program(["CYLINDER", "NUM_5", "EXTRUDE", "NUM_10", "BOX"])
        restored = CadProgram.from_json(program.to_json())
        assert restored == program

    def test_json_is_stable_across_runs(self):
        tokens = ["SKETCH_RECT", "NUM_80", "EXTRUDE", "NUM_10", "BOX"]
        assert parse_program(tokens).to_json() == parse_program(tokens).to_json()

    def test_schema_version_is_declared(self):
        program = parse_program(["BOX"])
        assert program.schema_version == CAD_IR_SCHEMA_VERSION


class TestIds:
    def test_program_id_is_deterministic(self):
        tokens = ["BOX", "EXTRUDE", "NUM_10"]
        assert parse_program(tokens).program_id == parse_program(tokens).program_id

    def test_program_id_changes_with_content(self):
        assert parse_program(["BOX"]).program_id != parse_program(["CYLINDER"]).program_id

    def test_op_ids_are_unique_and_stable(self):
        program = parse_program(["BOX", "EXTRUDE", "NUM_10"])
        ids = [s.op_id for s in program.steps]
        assert len(set(ids)) == len(ids)
        assert parse_program(["BOX", "EXTRUDE", "NUM_10"]).steps[1].op_id == ids[1]

    def test_ids_are_short_and_sha256_prefixed(self):
        program = parse_program(["BOX"])
        assert program.program_id.startswith("program:")
        assert len(program.program_id) == len("program:") + 12


class TestSchemaVersioning:
    def test_current_version_compatible(self):
        assert is_schema_compatible("1.0.0")

    def test_future_minor_and_patch_incompatible(self):
        assert not is_schema_compatible("1.1.0")
        assert not is_schema_compatible("1.0.1")

    def test_major_mismatch_incompatible(self):
        assert not is_schema_compatible("2.0.0")
        assert not is_schema_compatible("0.9.0")

    def test_malformed_version_incompatible(self):
        assert not is_schema_compatible("banana")
        assert not is_schema_compatible("")


class TestDatasetIntegration:
    def test_every_dataset_record_passes_ir_gate(self):
        records = build_synthetic_records(200, seed=7)
        assert records
        for record in records:
            tokens = record["cad"]
            assert validate_cad_program(tokens), tokens
            assert parse_program(tokens).to_tokens() == tokens

    def test_dataset_programs_have_unique_ids_for_unique_content(self):
        records = build_synthetic_records(150, seed=7)
        by_tokens: dict[str, str] = {}
        for record in records:
            tokens = tuple(record["cad"])
            program = parse_program(list(tokens))
            prior = by_tokens.get(tokens)
            if prior is None:
                by_tokens[tokens] = program.program_id
            else:
                assert program.program_id == prior, "same content, different ID"

    def test_dataset_records_serialize(self):
        records = build_synthetic_records(50, seed=7)
        for record in records:
            program = parse_program(record["cad"])
            assert CadProgram.from_json(program.to_json()).to_tokens() == record["cad"]

    def test_ir_gate_does_not_narrow_legacy_acceptance(self):
        from cadgenesis.execution.geometry_validation import validate_program

        records = build_synthetic_records(300, seed=42)
        for record in records:
            assert validate_program(record["cad"]), record["cad"]


class TestExecutionIntegration:
    def test_freecad_engine_executes_cad_program(self):
        from cadgenesis.execution.freecad_engine import FreeCADEngine

        program = parse_program(["SKETCH_RECT", "NUM_80", "EXTRUDE", "NUM_10", "BOX"])
        result = FreeCADEngine().execute(program)
        assert result["status"] == "ok"
        assert result["solid_count"] == 1

    def test_opencascade_engine_executes_cad_program(self):
        from cadgenesis.execution.opencascade_engine import OpenCascadeEngine

        program = parse_program(["CYLINDER", "NUM_5", "EXTRUDE", "NUM_10", "BOX"])
        result = OpenCascadeEngine().execute(program)
        assert result["status"] == "ok"

    def test_execution_engine_accepts_cad_program(self):
        from cadgenesis.execution.execution_engine import CADExecutionEngine

        program = parse_program(["BOX", "EXTRUDE", "NUM_10"])
        result = CADExecutionEngine().execute_and_evaluate(program)
        assert result.is_valid_geometry
        assert result.estimated_cost_usd == 25.0

    def test_execution_engine_parametric_json_records_program_id(self):
        from cadgenesis.execution.execution_engine import CADExecutionEngine

        program = parse_program(["BOX", "EXTRUDE", "NUM_10"])
        result = CADExecutionEngine().execute_and_evaluate(program)
        assert result.parametric_json["program_id"] == program.program_id
