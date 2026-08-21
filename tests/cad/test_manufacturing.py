"""Tests for cadgenesis.cad.manufacturing (features, process selection)."""

from __future__ import annotations

from cadgenesis.cad.manufacturing.features import (
    ManufacturingFeature,
    cnc_feature,
    make_feature,
    print_feature,
)
from cadgenesis.cad.manufacturing.process import (
    ProcessSelection,
    ProcessSelector,
)


class TestManufacturingFeatures:
    def test_cnc_feature(self) -> None:
        feature = cnc_feature("pocket", depth=5.0, tool_diameter=6.0)
        assert isinstance(feature, ManufacturingFeature)
        assert feature.process_group == "cnc"
        assert feature.operation == "pocket"

    def test_print_feature(self) -> None:
        feature = print_feature("fins", thickness=1.0)
        assert feature.process_group == "3d_printing"
        assert feature.params["thickness"] == 1.0

    def test_make_feature(self) -> None:
        feature = make_feature("casting", "housing", {"material": "AL-6061"})
        assert feature.process_group == "casting"

    def test_to_dict_roundtrip(self) -> None:
        feature = cnc_feature("hole", depth=10.0)
        restored = ManufacturingFeature.from_dict(feature.to_dict())
        assert restored.operation == feature.operation


class TestProcessSelector:
    def test_select_plastic(self) -> None:
        selector = ProcessSelector()
        selection = selector.select(
            {
                "material_category": "plastic",
                "batch_size": 5000,
                "max_part_size_mm": 80.0,
            }
        )
        assert isinstance(selection, ProcessSelection)
        assert selection.suggestions
        assert selection.best is not None

    def test_by_group(self) -> None:
        selector = ProcessSelector()
        selection = selector.select({"material_category": "metal", "batch_size": 10})
        grouped = selection.by_group()
        assert grouped

    def test_required_group(self) -> None:
        selector = ProcessSelector()
        selection = selector.select(
            {
                "material_category": "metal",
                "batch_size": 100,
                "required_group": "cnc",
            }
        )
        for suggestion in selection.suggestions:
            assert suggestion.group == "cnc"

    def test_suggestion_score_range(self) -> None:
        selector = ProcessSelector()
        selection = selector.select({"material_category": "metal", "batch_size": 100})
        for suggestion in selection.suggestions:
            assert 0.0 <= suggestion.score <= 1.0
