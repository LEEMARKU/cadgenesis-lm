"""tests/tokenizer/test_tokens.py
================================
Unit tests for the CAD token-definition modules.
"""

from __future__ import annotations

from cadgenesis.tokenizer import cad_tokens
from cadgenesis.tokenizer.assembly_tokens import ALL_ASSEMBLY_TOKENS
from cadgenesis.tokenizer.constraint_tokens import ALL_CONSTRAINT_TOKENS
from cadgenesis.tokenizer.feature_tokens import ALL_FEATURE_TOKENS
from cadgenesis.tokenizer.geometry_tokens import ALL_GEOMETRY_TOKENS
from cadgenesis.tokenizer.manufacturing_tokens import ALL_MANUFACTURING_TOKENS
from cadgenesis.tokenizer.material_tokens import ALL_MATERIAL_TOKENS
from cadgenesis.tokenizer.simulation_tokens import ALL_SIMULATION_TOKENS


class TestFamilyTables:
    def test_all_families_nonempty(self):
        for table in [
            ALL_GEOMETRY_TOKENS,
            ALL_FEATURE_TOKENS,
            ALL_CONSTRAINT_TOKENS,
            ALL_MATERIAL_TOKENS,
            ALL_ASSEMBLY_TOKENS,
            ALL_MANUFACTURING_TOKENS,
            ALL_SIMULATION_TOKENS,
        ]:
            assert len(table) > 0
            assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in table)

    def test_known_geometry_tokens(self):
        names = {t for t, _ in ALL_GEOMETRY_TOKENS}
        assert "PRIM_BOX" in names
        assert "CURVE_LINE" in names
        assert "BREP_EDGE" in names

    def test_known_feature_tokens(self):
        names = {t for t, _ in ALL_FEATURE_TOKENS}
        assert "EXTRUDE" in names or "FEAT_EXTRUDE" in names
        assert any("FILLET" in n for n in names)

    def test_known_simulation_tokens(self):
        names = {t for t, _ in ALL_SIMULATION_TOKENS}
        assert len(names) > 0


class TestSpecialTokens:
    def test_controls_present(self):
        joined = "".join(cad_tokens.SPECIAL_CAD_TOKENS)
        assert "<pad>" in joined
        assert "<bos>" in joined
        assert "<eos>" in joined
        assert "<unk>" in joined
        assert len(cad_tokens.SPECIAL_CAD_TOKENS) >= 10

    def test_family_tables_mapping(self):
        assert set(cad_tokens.FAMILY_TOKEN_TABLES) == {
            "geometry",
            "feature",
            "constraint",
            "material",
            "assembly",
            "manufacturing",
            "simulation",
        }


class TestAggregate:
    def test_static_table_matches_families(self):
        total = (
            len(ALL_GEOMETRY_TOKENS)
            + len(ALL_FEATURE_TOKENS)
            + len(ALL_CONSTRAINT_TOKENS)
            + len(ALL_MATERIAL_TOKENS)
            + len(ALL_ASSEMBLY_TOKENS)
            + len(ALL_MANUFACTURING_TOKENS)
            + len(ALL_SIMULATION_TOKENS)
        )
        assert len(cad_tokens.STATIC_CAD_TOKEN_TABLES) == total

    def test_all_tables_includes_numeric(self):
        all_tables = cad_tokens.all_cad_token_tables()
        assert len(all_tables) > len(cad_tokens.STATIC_CAD_TOKEN_TABLES)
        names = {t for t, _ in all_tables}
        assert any(n.startswith("NUM_") for n in names)

    def test_all_strings_unique(self):
        names = cad_tokens.all_cad_token_strings()
        assert len(names) == len(set(names))

    def test_roundtrip_generated_numeric(self):
        from cadgenesis.tokenizer.numeric_tokens import (
            all_numeric_token_strings,
            build_numeric_token_table,
        )

        table = build_numeric_token_table()
        assert len(table) == len(all_numeric_token_strings())
        assert len(table) > 100
