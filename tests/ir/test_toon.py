"""
tests/ir/test_toon.py
=====================
v6.3 TOON bridge: IR <-> TOON grammar round-trips and validator parity with
the critique engine + tokenizer gate.
"""

from __future__ import annotations

from cadgenesis.distillation.critique import CritiqueEngine
from cadgenesis.ir import (
    TOON_FIELDS,
    toon_program_is_valid,
    toon_to_program,
    validate_toon_program,
)
from cadgenesis.ir.toon import program_to_toon
from cadgenesis.tokenizer import AutonomousCADTokenizer

GOOD_TOON = (
    "id|feature|width|height|depth|fillet\n"
    "int|str|float|float|float|float\n"
    "1|BOX|50.0|30.0|20.0|2.0\n"
    "2|CYLINDER|10.0|40.0|10.0|0.5"
)


class TestToonToProgram:
    def test_parses_rows_into_ops(self):
        program = toon_to_program(GOOD_TOON)
        assert len(program.steps) == 2
        assert [s.kind for s in program.steps] == ["PRIM_BOX", "PRIM_CYLINDER"]
        assert program.steps[0].params["d0"] == 50.0
        assert program.steps[0].params["d1"] == 30.0
        assert program.steps[0].params["d2"] == 20.0
        assert program.steps[0].params["fillet"] == 2.0

    def test_dependency_chain(self):
        program = toon_to_program(GOOD_TOON)
        assert program.steps[0].depends_on == ()
        assert program.steps[1].depends_on == (program.steps[0].op_id,)

    def test_deterministic_ids(self):
        assert toon_to_program(GOOD_TOON).program_id == toon_to_program(GOOD_TOON).program_id

    def test_empty_toon_is_empty_program(self):
        assert toon_to_program("").steps == ()

    def test_unknown_feature_becomes_raw(self):
        program = toon_to_program(
            "id|feature\nint|str\n1|TORUS_WIDGET"
        )
        assert program.steps[0].kind == "RAW"

    def test_missing_dims_are_absent_params(self):
        program = toon_to_program("id|feature\nint|str\n1|SPHERE")
        assert "d0" not in program.steps[0].params


class TestProgramToToon:
    def test_round_trip_is_lossless(self):
        program = toon_to_program(GOOD_TOON)
        report = program_to_toon(program)
        assert report.fully_mapped
        assert toon_to_program(report.toon).to_tokens() == program.to_tokens()
        assert report.toon == GOOD_TOON

    def test_header_and_schema_line(self):
        program = toon_to_program(GOOD_TOON)
        report = program_to_toon(program)
        lines = report.toon.splitlines()
        assert lines[0] == "|".join(TOON_FIELDS)
        assert lines[1] == "int|str|float|float|float|float"

    def test_unmappable_kinds_are_reported_not_dropped(self):
        from cadgenesis.ir.parser import parse_program

        program = parse_program(["BOX", "FEAT_HOLE", "NUM_010"])
        report = program_to_toon(program)
        assert not report.fully_mapped
        assert report.skipped == ["FEAT_HOLE"]
        assert "|".join(TOON_FIELDS) in report.toon

    def test_fillet_preserved_through_round_trip(self):
        program = toon_to_program(GOOD_TOON)
        restored = toon_to_program(program_to_toon(program).toon)
        assert restored.steps[0].params["fillet"] == 2.0


class TestCanonicalTokensClassify:
    def test_registered_canonical_tokens_open_ops(self):
        from cadgenesis.ir import canonical_kind, is_feature_kind, is_feature_token

        assert canonical_kind("FEAT_PATTERN_LIN") == "FEAT_PATTERN_LIN"
        assert canonical_kind("FEAT_BOOL_UNION") == "FEAT_BOOL_UNION"
        assert canonical_kind("FEAT_HOLE_CB") == "FEAT_HOLE_CB"
        assert is_feature_kind("FEAT_PATTERN_LIN")
        assert is_feature_token("FEAT_PATTERN_LIN")
        assert canonical_kind("PRIM_CAPSULE") == "PRIM_CAPSULE"

    def test_canonical_feature_program_parses_into_steps(self):
        from cadgenesis.ir import parse_program, validate_cad_program

        tokens = ["PRIM_BOX", "NUM_050", "FEAT_FILLET", "NUM_005", "FEAT_HOLE", "NUM_010"]
        program = parse_program(tokens)
        kinds = [s.kind for s in program.steps]
        assert kinds == ["PRIM_BOX", "FEAT_FILLET", "FEAT_HOLE"]
        assert program.to_tokens() == tokens
        assert validate_cad_program(tokens)


class TestToonValidation:
    def test_good_toon_passes(self):
        report = validate_toon_program(GOOD_TOON)
        assert report.passed
        assert all(c.passed for c in report.checks)

    def test_gate_matches_report(self):
        assert toon_program_is_valid(GOOD_TOON)

    def test_negative_dimension_rejected(self):
        bad = GOOD_TOON.replace("50.0|30.0", "-5.0|30.0")
        report = validate_toon_program(bad)
        assert not report.passed
        assert any(c.name == "toon_dims_positive" and not c.passed for c in report.checks)

    def test_zero_dimension_rejected(self):
        bad = GOOD_TOON.replace("50.0|30.0", "0.0|30.0")
        assert not toon_program_is_valid(bad)

    def test_non_numeric_dimension_rejected(self):
        bad = GOOD_TOON.replace("50.0|30.0", "big|30.0")
        report = validate_toon_program(bad)
        assert any(c.name == "toon_dims_numeric" and not c.passed for c in report.checks)

    def test_fillet_exceeding_ratio_rejected(self):
        bad = GOOD_TOON.replace("20.0|2.0\n2|CYLINDER", "20.0|30.0\n2|CYLINDER")
        report = validate_toon_program(bad)
        assert any(c.name == "toon_fillet_ratio" and not c.passed for c in report.checks)

    def test_negative_fillet_rejected(self):
        bad = GOOD_TOON.replace("|2.0\n", "|-2.0\n")
        assert not toon_program_is_valid(bad)

    def test_empty_feature_rejected(self):
        bad = GOOD_TOON.replace("1|BOX", "1|")
        report = validate_toon_program(bad)
        assert any(c.name == "toon_features" and not c.passed for c in report.checks)

    def test_unknown_feature_rejected(self):
        bad = GOOD_TOON.replace("2|CYLINDER", "2|TORUS_WIDGET")
        report = validate_toon_program(bad)
        assert any(c.name == "toon_features" and not c.passed for c in report.checks)

    def test_toon_parse_is_total_and_gate_still_fails(self):
        """The TOON parser is total (never raises): garbage yields rows that
        fail the feature/dimension checks, and only an empty payload trips
        ``toon_parse`` itself."""
        report = validate_toon_program("not|a|toon|row\n~~~")
        assert any(c.name == "toon_parse" and c.passed for c in report.checks)
        assert not report.passed
        assert any(c.name == "toon_features" and not c.passed for c in report.checks)

    def test_empty_payload_rejected(self):
        assert not toon_program_is_valid("")

    def test_parity_with_critique_engine(self):
        """The IR gate must agree with the critique engine on the same payloads."""
        from cadgenesis.distillation.critique import _POSITIVE_DIMENSION_KEYS
        from cadgenesis.ir.toon_validation import POSITIVE_DIMENSION_KEYS

        # rule parity: same positive-dimension key set, same fillet ratio
        assert POSITIVE_DIMENSION_KEYS == _POSITIVE_DIMENSION_KEYS

        critique = CritiqueEngine()
        cases = [
            GOOD_TOON,
            GOOD_TOON.replace("50.0|30.0", "-5.0|30.0"),
            GOOD_TOON.replace("20.0|2.0\n2|CYLINDER", "20.0|30.0\n2|CYLINDER"),
            GOOD_TOON.replace("1|BOX", "1|"),
        ]
        for toon in cases:
            crit = critique.critique(toon, prompt="make a box")
            assert crit.score == 1.0 if crit.issues == [] else crit.score < 1.0
            # critique score 1.0  <=>  zero issues  <=>  IR gate passes
            assert toon_program_is_valid(toon) == (crit.score == 1.0), toon


class TestVocabAwareValidation:
    def test_default_vocab_registers_legacy_tokens(self):
        tok = AutonomousCADTokenizer.build()
        from cadgenesis.ir import parse_program, validate_program_ir

        program = parse_program(["SKETCH_RECT", "NUM_80", "EXTRUDE", "NUM_10", "BOX"])
        report = validate_program_ir(program, vocab=tok.vocab)
        assert report.passed
        assert any(c.name == "tokens_registered" and c.passed for c in report.checks)

    def test_mini_vocab_rejects_default_only_tokens(self):
        tok = AutonomousCADTokenizer.build_mini()
        from cadgenesis.ir import parse_program, validate_program_ir

        program = parse_program(["BOX", "FEAT_HOLE", "NUM_010"])
        report = validate_program_ir(program, vocab=tok.vocab)
        assert not report.passed
        check = next(c for c in report.checks if c.name == "tokens_registered")
        assert not check.passed
        assert "FEAT_HOLE" in check.detail

    def test_mini_vocab_accepts_mini_program(self):
        tok = AutonomousCADTokenizer.build_mini()
        from cadgenesis.ir import parse_program, validate_program_ir

        program = parse_program(["EXTRUDE", "NUM_10", "BOX"])
        assert validate_program_ir(program, vocab=tok.vocab).passed

    def test_no_vocab_skips_registered_check(self):
        from cadgenesis.ir import validate_program_ir

        program = toon_to_program(GOOD_TOON)
        report = validate_program_ir(program)
        assert report.passed
        assert all(c.name != "tokens_registered" for c in report.checks)