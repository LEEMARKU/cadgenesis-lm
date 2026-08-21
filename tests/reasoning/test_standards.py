"""Tests for Pillar 7 engineering standards engine."""

from __future__ import annotations

import pytest

from cadgenesis.reasoning import (
    Standard,
    StandardsCheck,
    StandardsLibrary,
    build_standards_graph,
    default_standards_library,
)
from cadgenesis.reasoning.standards import STANDARD_BODIES


def test_standard_validation() -> None:
    with pytest.raises(ValueError):
        Standard("UNKNOWN", "X-1")
    with pytest.raises(ValueError):
        Standard("ISO", "")
    with pytest.raises(TypeError):
        Standard("ISO", "X", check="not callable")


def test_default_library_shape() -> None:
    library = default_standards_library()
    assert len(library) >= 10
    for body in ("ISO", "ASME", "DIN", "ANSI", "COMPANY"):
        assert body in library.bodies


def test_tolerance_lookup() -> None:
    library = default_standards_library()
    assert library.tolerance(25.0, 7) == 21
    assert library.tolerance(400.0, 6) == 40
    assert library.tolerance(100.0, 14) is None
    with pytest.raises(ValueError):
        library.tolerance(-1.0, 7)


def test_fit_and_roughness_lookups() -> None:
    library = default_standards_library()
    assert library.fit("rc3") == (12.0, 44.0)
    assert library.fit("LN2") == (-24.0, -8.0)
    assert library.fit("ZZ9") is None
    assert library.roughness_grade("N7") == 1.6
    assert library.roughness_grade("n7") == 1.6


def test_thread_and_material_lookups() -> None:
    library = default_standards_library()
    assert library.thread_pitch("M10") == 1.5
    assert library.thread_pitch("M99") is None
    material = library.material("AISI 316")
    assert material is not None
    assert material["family"] == "steel"


def test_compliance_checks_apply_body_filter() -> None:
    library = default_standards_library()
    part = {"kind": "tolerance", "grade": 7, "tolerance_um": 25.0, "standards": ["ISO"]}
    results = library.compliance(part)
    assert results
    assert all(isinstance(r, StandardsCheck) for r in results)
    assert all(r.standard.body == "ISO" for r in results)
    assert all(r.passed for r in results)


def test_compliance_fails_rule() -> None:
    library = default_standards_library()
    part = {"sharp_internal_corners": True}
    results = library.compliance(part)
    failed = [r for r in results if not r.passed]
    assert any("no_sharp_internal_corners" in r.standard.identifier for r in failed)


def test_passed_and_summary() -> None:
    library = default_standards_library()
    part = {"grade": 7, "tolerance_um": 25.0, "min_edge_radius": 0.5}
    assert library.passed(part)
    summary = library.summary()
    assert summary["total"] == len(library)
    assert "ISO" in summary["bodies"]


def test_registry_duplicates_and_removal() -> None:
    library = StandardsLibrary([Standard("ISO", "ISO-X", kind="rule")])
    with pytest.raises(ValueError):
        library.register(Standard("ISO", "ISO-X", kind="rule"))
    assert library.get("ISO-X") is not None
    assert library.by_body("iso")[0].identifier == "ISO-X"


def test_custom_standard_check() -> None:
    library = StandardsLibrary(
        [Standard("COMPANY", "max_weight", kind="rule", check=lambda p: p.get("weight", 0) <= 10)]
    )
    assert library.passed({"weight": 5})
    assert not library.passed({"weight": 20})


def test_build_standards_graph() -> None:
    graph = build_standards_graph()
    assert graph.node_count >= len(default_standards_library())
    iso = graph.get_node("ISO")
    assert iso is not None and iso.node_type == "body"
    related = graph.find_related("ISO", max_depth=1)
    assert len(related) >= 1


def test_standard_bodies_constant() -> None:
    assert STANDARD_BODIES == ("ISO", "ASME", "DIN", "ANSI", "COMPANY")
