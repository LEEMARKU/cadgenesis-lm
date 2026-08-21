"""Tests for cadgenesis.cad.features (feature tree, operations, types)."""

from __future__ import annotations

from cadgenesis.cad.features import (
    BooleanSubtract,
    BooleanUnion,
    Chamfer,
    CircularPattern,
    Fillet,
    LinearPattern,
    Loft,
    Revolve,
    Sweep,
)
from cadgenesis.cad.features.base import (
    FEATURE_REGISTRY,
    FeatureTree,
    FeatureType,
    create_feature,
    known_feature_types,
)
from cadgenesis.cad.features.solids import Extrude


class TestFeatureTypes:
    def test_enum_members(self) -> None:
        expected = {
            "EXTRUDE",
            "REVOLVE",
            "LOFT",
            "SWEEP",
            "FILLET",
            "CHAMFER",
            "SHELL",
            "HOLE",
            "LINEAR_PATTERN",
            "BOOLEAN_UNION",
            "BOOLEAN_SUBTRACT",
            "BOOLEAN_INTERSECT",
        }
        assert expected.issubset({t.value for t in FeatureType})

    def test_registry_has_extrude(self) -> None:
        assert "EXTRUDE" in FEATURE_REGISTRY

    def test_create_feature(self) -> None:
        feature = create_feature("EXTRUDE", name="base", params={"depth": 10.0})
        assert feature.type == FeatureType.EXTRUDE

    def test_known_types(self) -> None:
        assert "FILLET" in known_feature_types()


class TestExtrude:
    def test_basic(self) -> None:
        extrude = Extrude(name="e", sketch_ref="sketch1", params={"depth": 5.0})
        assert extrude.params["depth"] == 5.0
        assert extrude.type == FeatureType.EXTRUDE

    def test_validate_requires_sketch(self) -> None:
        extrude = Extrude(name="e", params={"depth": 5.0})
        assert any("requires a sketch profile" in p for p in extrude.validate())


class TestFeatureTree:
    def test_add_and_get(self) -> None:
        tree = FeatureTree()
        feature = create_feature("EXTRUDE", name="e1")
        tree.add(feature)
        assert tree.get("e1") is feature

    def test_execution_order(self) -> None:
        tree = FeatureTree()
        tree.add(create_feature("EXTRUDE", name="e1"))
        tree.add(create_feature("FILLET", name="f1"))
        order = tree.execution_order()
        assert order[0].name == "e1"
        assert order[-1].name == "f1"

    def test_feature_count(self) -> None:
        tree = FeatureTree()
        tree.add(create_feature("EXTRUDE", name="e1"))
        tree.add(create_feature("SHELL", name="s1"))
        assert len(tree) == 2


class TestOtherFeatures:
    def test_revolve(self) -> None:
        feature = Revolve(name="r", sketch_ref="s1", params={"angle": 90.0})
        assert feature.type == FeatureType.REVOLVE
        assert feature.params["angle"] == 90.0

    def test_sweep(self) -> None:
        feature = Sweep(name="sw", sketch_ref="s1", params={"path": "edge1"})
        assert feature.type == FeatureType.SWEEP

    def test_loft(self) -> None:
        feature = Loft(name="l", params={"sections": ["s1", "s2"]})
        assert feature.type == FeatureType.LOFT

    def test_linear_pattern(self) -> None:
        feature = LinearPattern(
            name="lp", params={"count": 4, "spacing": 10.0}, references=["base"]
        )
        assert feature.type == FeatureType.LINEAR_PATTERN
        assert feature.params["count"] == 4

    def test_circular_pattern(self) -> None:
        CircularPattern(name="cp", params={"count": 6, "axis": "z"}, references=["base"])
        assert "CIRCULAR_PATTERN" in {t.value for t in FeatureType}

    def test_boolean_union(self) -> None:
        feature = BooleanUnion(name="bu", params={"operation": "union"}, references=["a", "b"])
        assert feature.type == FeatureType.BOOLEAN_UNION

    def test_boolean_subtract(self) -> None:
        feature = BooleanSubtract(
            name="bs", params={"operation": "subtract"}, references=["a", "b"]
        )
        assert feature.type == FeatureType.BOOLEAN_SUBTRACT

    def test_fillet_requires_edge(self) -> None:
        feature = Fillet(name="f", params={"radius": 2.0})
        assert any("edge reference" in p for p in feature.validate())

    def test_chamfer_requires_edge(self) -> None:
        feature = Chamfer(name="c", params={"distance": 1.0})
        assert feature.validate()
