"""Tests for cadgenesis.cad.materials and cadgenesis.cad.gdt."""

from __future__ import annotations

import pytest

from cadgenesis.cad.gdt import (
    Datum,
    DatumReference,
    FeatureControlFrame,
    GDTSpecification,
    ManufacturingTolerance,
)
from cadgenesis.cad.materials.database import MATERIALS, Material, MaterialDatabase


class TestMaterials:
    def test_database_lookup(self) -> None:
        db = MaterialDatabase()
        material = db.get("AISI 1045")
        assert isinstance(material, Material)
        assert material.category == "metal"

    def test_alias_lookup(self) -> None:
        db = MaterialDatabase()
        assert "abs" in db
        assert db["ABS"].name  # type: ignore[union-attr]

    def test_property_access(self) -> None:
        material = MaterialDatabase().get("AISI 1045")
        assert material.density() > 0
        assert material.youngs_modulus() > 0

    def test_categories(self) -> None:
        db = MaterialDatabase()
        assert db.metals()
        assert db.plastics()
        assert db.composites()
        assert db.ceramics()

    def test_unknown_material(self) -> None:
        with pytest.raises(KeyError):
            MaterialDatabase().get("not-a-material")

    def test_builtin_materials_populated(self) -> None:
        assert len(MATERIALS) > 10

    def test_invalid_category(self) -> None:
        with pytest.raises(ValueError):
            Material(name="x", category="unknown", properties={})


class TestGDT:
    def test_datum_reference(self) -> None:
        ref = DatumReference("A")
        assert ref.datum == "A"

    def test_control_frame(self) -> None:
        frame = FeatureControlFrame(characteristic="POSITION", tolerance=0.05)
        assert frame.class_name  # resolves to a non-empty tolerance class name

    def test_invalid_characteristic(self) -> None:
        with pytest.raises(ValueError):
            FeatureControlFrame(characteristic="NOT_REAL", tolerance=0.05)

    def test_invalid_tolerance(self) -> None:
        with pytest.raises(ValueError):
            FeatureControlFrame(characteristic="FLATNESS", tolerance=0.0)

    def test_spec_valid(self) -> None:
        spec = GDTSpecification(
            datums=[Datum(identifier="A")],
            control_frames=[
                FeatureControlFrame(
                    characteristic="POSITION",
                    tolerance=0.1,
                    datums=[DatumReference("A")],
                )
            ],
        )
        assert spec.validate() == []

    def test_spec_undefined_datum(self) -> None:
        spec = GDTSpecification(
            control_frames=[
                FeatureControlFrame(
                    characteristic="POSITION",
                    tolerance=0.1,
                    datums=[DatumReference("Z")],
                )
            ]
        )
        assert spec.validate()

    def test_manufacturing_tolerance(self) -> None:
        tolerance = ManufacturingTolerance(
            kind="LIMIT", lower_limit=9.95, upper_limit=10.05, feature="bore"
        )
        assert tolerance.feature == "bore"

    def test_invalid_limit_tolerance(self) -> None:
        spec = GDTSpecification(
            manufacturing_tolerances=[
                ManufacturingTolerance(kind="LIMIT", lower_limit=10.5, upper_limit=10.0)
            ]
        )
        assert spec.validate()

    def test_dict_roundtrip(self) -> None:
        spec = GDTSpecification(
            datums=[Datum(identifier="A")],
            control_frames=[FeatureControlFrame(characteristic="FLATNESS", tolerance=0.02)],
        )
        restored = GDTSpecification.from_dict(spec.to_dict())
        assert restored.to_dict() == spec.to_dict()
