"""
tests/tokenizer/test_vocab_separation.py
========================================
Text / CAD vocabulary separation invariant (v6.1 §4.5).

The tokenizer's single flat ID space must never let language tokens collide
with CAD tokens: the LANGUAGE family always starts exactly at the model's
``cad_vocab_size`` (the sum of the non-language slot capacities), for both
the default and the 1024-slot mini vocabularies.
"""

from __future__ import annotations

import pytest

from cadgenesis.config import CADConfig
from cadgenesis.tokenizer import AutonomousCADTokenizer
from cadgenesis.tokenizer.vocabulary import TokenFamily
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer


def _language_start(tok: AutonomousCADTokenizer) -> int:
    return tok.vocab._ranges[TokenFamily.LANGUAGE].start


def _model_cad_vocab_size(cfg: CADConfig) -> int:
    return GeometryAwareTransformer(cfg).cad_vocab_size


@pytest.mark.parametrize(
    "cfg_factory, tok_factory",
    [
        (CADConfig.mini, AutonomousCADTokenizer.build_mini),
        (CADConfig, AutonomousCADTokenizer.build),
    ],
    ids=["mini", "default"],
)
def test_language_range_starts_at_cad_vocab_size(cfg_factory, tok_factory):
    """CAD ids and language ids must occupy disjoint, adjacent ranges."""
    tok = tok_factory()
    model_cad = _model_cad_vocab_size(cfg_factory())
    lang_start = _language_start(tok)

    assert lang_start == model_cad, (
        f"language family starts at {lang_start} but the model's CAD "
        f"vocabulary has {model_cad} ids — overlap or gap between the two spaces."
    )


@pytest.mark.parametrize(
    "cfg_factory, tok_factory",
    [
        (CADConfig.mini, AutonomousCADTokenizer.build_mini),
        (CADConfig, AutonomousCADTokenizer.build),
    ],
    ids=["mini", "default"],
)
def test_no_cad_token_maps_into_language_range(cfg_factory, tok_factory):
    """Every registered CAD token must live strictly below the language range."""
    tok = tok_factory()
    lang_start = _language_start(tok)
    for record in tok.vocab:
        if record.family is TokenFamily.LANGUAGE:
            assert record.token_id >= lang_start
        else:
            assert record.token_id < lang_start, (
                f"CAD token {record.token_str!r} (id {record.token_id}) collides "
                f"with the language range starting at {lang_start}."
            )


def test_mini_is_exactly_1024_slots():
    """The mini layout is a 1024-slot vocabulary (v6.1 §4.4)."""
    cfg = CADConfig.mini()
    t = cfg.tokenizer
    cad_slots = (
        t.geometry_token_slots
        + t.feature_token_slots
        + t.constraint_token_slots
        + t.material_token_slots
        + t.assembly_token_slots
        + t.manufacturing_token_slots
        + t.simulation_token_slots
        + t.numeric_token_slots
        + t.special_token_slots
    )
    assert cad_slots + t.lang_vocab_size == 1024
    tok = AutonomousCADTokenizer.build_mini()
    assert tok.vocab._ranges[TokenFamily.LANGUAGE].start == cad_slots == 512
    assert GeometryAwareTransformer(cfg).cad_vocab_size == 512