"""tests/transformer/evolution/test_versioning.py
===================================================
Unit tests for architecture versioning and content hashing.
"""

from __future__ import annotations

import pytest

from cadgenesis.transformer.evolution.versioning import (
    ArchitectureVersion,
    VersionedArchitecture,
    hash_architecture,
)


class TestHashArchitecture:
    def test_hash_deterministic(self):
        a = hash_architecture({"a": 1, "b": [2, 3]})
        b = hash_architecture({"b": [2, 3], "a": 1})
        assert a == b
        assert len(a) == 64

    def test_hash_rejects_non_dict(self):
        with pytest.raises(TypeError):
            hash_architecture([1, 2])


class TestArchitectureVersion:
    def test_version_bump(self):
        v = ArchitectureVersion(1, 2, 3)
        assert str(v.bump("major")) == "2.0.0"
        assert str(v.bump("minor")) == "1.3.0"
        assert str(v.bump("patch")) == "1.2.4"

    def test_version_parse(self):
        assert str(ArchitectureVersion.parse("3.4.5")) == "3.4.5"
        with pytest.raises(ValueError):
            ArchitectureVersion.parse("3.4")

    def test_invalid_part(self):
        with pytest.raises(ValueError):
            ArchitectureVersion(1, 0, 0).bump("unknown")


class TestVersionedArchitecture:
    def test_upgrade(self):
        va = VersionedArchitecture("hier", {"layers": 3})
        hash1 = va.content_hash
        va.upgrade({"layers": 4}, bump="minor")
        assert va.version.minor == 1
        assert va.content_hash != hash1
        assert len(va.history) == 1
        assert va.full_version().startswith("hier@1.1.0")

    def test_roundtrip(self, tmp_path):
        va = VersionedArchitecture("hier", {"layers": 3}, version="2.1.0")
        path = tmp_path / "arch.json"
        va.save(str(path))
        loaded = VersionedArchitecture.load(str(path))
        assert loaded.name == "hier"
        assert str(loaded.version) == "2.1.0"
        assert loaded.spec == {"layers": 3}

    def test_upgrade_rejects_non_dict(self):
        va = VersionedArchitecture("hier", {"layers": 3})
        with pytest.raises(TypeError):
            va.upgrade([1, 2])

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError):
            VersionedArchitecture("  ", {"a": 1})

    def test_fingerprint(self):
        va = VersionedArchitecture("hier", {"layers": 3}, version="1.2.0")
        fp = va.fingerprint()
        assert fp["name"] == "hier"
        assert fp["version"] == "1.2.0"
        assert len(fp["hash"]) == 64
