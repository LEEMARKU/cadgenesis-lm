"""tests/memory/test_cad_memory.py
=================================
Unit tests for the CAD memory pool.
"""

from __future__ import annotations

from cadgenesis.memory.cad_memory import CADMemory


def test_remember_feature_tree():
    store = CADMemory(capacity=16)
    store.remember_feature_tree("part:base", [{"op": "extrude"}], kind="part")
    assert store.recall("extrude")[0].entry.key == "part:base"
    assert store.recall("extrude", object_kind="part")[0].entry.key == "part:base"
    assert store.recall("extrude", object_kind="bracket") == []


def test_remember_brep():
    store = CADMemory(capacity=16)
    store.remember_brep("part:brim", {"shells": 1}, kind="bracket")
    assert store.by_kind("bracket")[0].key == "part:brim"


def test_by_kind():
    store = CADMemory(capacity=16)
    store.remember_feature_tree("a", [{"op": "extrude"}], kind="part")
    store.remember_feature_tree("b", [{"op": "fillet"}], kind="part")
    store.remember_feature_tree("c", [{"op": "loft"}], kind="body")
    assert len(store.by_kind("part")) == 2
    assert len(store.by_kind("body")) == 1
