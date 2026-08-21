"""Tests for cadgenesis.adapters.versioning."""

import pytest

from cadgenesis.adapters.versioning import AdapterVersion, AdapterVersionRegistry


def test_from_string_variants():
    assert AdapterVersion.from_string("1.2.3") == AdapterVersion(major=1, minor=2, patch=3)
    assert AdapterVersion.from_string("v1.2.3") == AdapterVersion(1, 2, 3)


def test_from_string_invalid_raises():
    with pytest.raises(ValueError):
        AdapterVersion.from_string("1.2")
    with pytest.raises(ValueError):
        AdapterVersion.from_string("x.y.z")
    with pytest.raises(ValueError):
        AdapterVersion.from_string("")


def test_negative_component_raises():
    with pytest.raises(ValueError):
        AdapterVersion(major=-1, minor=0, patch=0)


def test_to_string():
    assert AdapterVersion(1, 2, 3).to_string() == "v1.2.3"
    assert AdapterVersion(0, 0, 0).to_string() == "v0.0.0"


def test_comparisons_and_sorting():
    assert AdapterVersion(1, 2, 3) < AdapterVersion(1, 2, 4)
    assert AdapterVersion(1, 2, 3) < AdapterVersion(1, 3, 0)
    assert AdapterVersion(1, 9, 9) < AdapterVersion(2, 0, 0)
    assert AdapterVersion(2, 0, 0) > AdapterVersion(1, 2, 3)
    versions = [
        AdapterVersion(1, 10, 0),
        AdapterVersion(1, 2, 3),
        AdapterVersion(2, 0, 0),
    ]
    assert sorted(versions) == [AdapterVersion(1, 2, 3), AdapterVersion(1, 10, 0), versions[2]]


def test_is_compatible():
    v1 = AdapterVersion(1, 2, 0)
    assert v1.is_compatible(AdapterVersion(1, 5, 0))
    assert not v1.is_compatible(AdapterVersion(2, 0, 0))
    assert not AdapterVersion(0, 1, 0).is_compatible(AdapterVersion(0, 2, 0), min_major=1)


def test_register_returns_unique_tag():
    registry = AdapterVersionRegistry()
    tag = registry.register("aero_v1", AdapterVersion(1, 0, 0))
    assert tag == "aero_v1@v1.0.0"
    tag2 = registry.register("aero_v1", AdapterVersion.from_string("v1.2.3"))
    assert tag2 == "aero_v1@v1.2.3"
    assert tag != tag2


def test_resolve():
    registry = AdapterVersionRegistry()
    registry.register("aero_v1", AdapterVersion(1, 0, 0))
    assert registry.resolve("aero_v1@v1.0.0") == AdapterVersion(1, 0, 0)


def test_resolve_unknown_raises():
    registry = AdapterVersionRegistry()
    registry.register("aero_v1", AdapterVersion(1, 0, 0))
    with pytest.raises(KeyError):
        registry.resolve("aero_v1@v9.9.9")
    with pytest.raises(ValueError):
        registry.resolve("notatag")
    with pytest.raises(ValueError):
        registry.resolve("aero_v1@v1.0")


def test_latest():
    registry = AdapterVersionRegistry()
    assert registry.latest("aero_v1") is None
    registry.register("aero_v1", AdapterVersion(1, 0, 0))
    registry.register("aero_v1", AdapterVersion(1, 0, 5))
    registry.register("aero_v1", AdapterVersion(1, 1, 0))
    assert registry.latest("aero_v1") == AdapterVersion(1, 1, 0)


def test_bump_sequences():
    registry = AdapterVersionRegistry()
    assert registry.bump("aero_v1") == AdapterVersion(1, 0, 0)
    assert registry.bump("aero_v1", "patch") == AdapterVersion(1, 0, 1)
    assert registry.bump("aero_v1", "minor") == AdapterVersion(1, 1, 0)
    assert registry.bump("aero_v1", "major") == AdapterVersion(2, 0, 0)
    assert registry.latest("aero_v1") == AdapterVersion(2, 0, 0)


def test_bump_invalid_part_raises():
    registry = AdapterVersionRegistry()
    registry.register("aero_v1", AdapterVersion(1, 0, 0))
    with pytest.raises(ValueError):
        registry.bump("aero_v1", "micro")
