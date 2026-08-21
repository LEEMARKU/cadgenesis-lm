from __future__ import annotations

import pytest

from cadgenesis.research.datasets import DatasetRegistry, bump_version, sha256_file


class TestHelpers:
    def test_sha256_file(self, tmp_path):
        file = tmp_path / "data.txt"
        file.write_text("hello", encoding="utf-8")
        assert sha256_file(file) == sha256_file(file)
        assert len(sha256_file(file)) == 64

    def test_bump_version(self):
        assert bump_version("1.2.3") == "1.2.4"
        assert bump_version("1.2.3", "minor") == "1.3.0"
        assert bump_version("1.2.3", "major") == "2.0.0"


class TestDatasetRegistry:
    def test_snapshot_file(self, tmp_path):
        source = tmp_path / "src.jsonl"
        source.write_text("line\n", encoding="utf-8")
        registry = DatasetRegistry(tmp_path / "datasets")
        version = registry.snapshot(name="cad", source=str(source))
        assert version.name == "cad"
        assert version.version == "0.0.1"
        assert registry.get("cad").version == "0.0.1"
        assert registry.verify("cad", "0.0.1") is True

    def test_auto_version_increment(self, tmp_path):
        source = tmp_path / "src.jsonl"
        source.write_text("line\n", encoding="utf-8")
        registry = DatasetRegistry(tmp_path / "datasets")
        registry.snapshot(name="cad", source=str(source))
        second = registry.snapshot(name="cad", source=str(source))
        assert second.version == "0.0.2"

    def test_explicit_version_and_duplicate(self, tmp_path):
        source = tmp_path / "src.jsonl"
        source.write_text("line\n", encoding="utf-8")
        registry = DatasetRegistry(tmp_path / "datasets")
        registry.snapshot(name="cad", source=str(source), version="2.1.0")
        with pytest.raises(ValueError):
            registry.snapshot(name="cad", source=str(source), version="2.1.0")

    def test_lineage(self, tmp_path):
        source = tmp_path / "src.jsonl"
        source.write_text("line\n", encoding="utf-8")
        registry = DatasetRegistry(tmp_path / "datasets")
        first = registry.snapshot(name="cad", source=str(source))
        second = registry.snapshot(name="cad", source=str(source))
        lineage = registry.lineage("cad", second.version)
        assert [v.version for v in lineage] == [second.version, first.version]  # newest first

    def test_rollback_alias(self, tmp_path):
        source = tmp_path / "src.jsonl"
        source.write_text("line\n", encoding="utf-8")
        registry = DatasetRegistry(tmp_path / "datasets")
        registry.snapshot(name="cad", source=str(source))
        registry.snapshot(name="cad", source=str(source))
        assert registry.get("cad").version == "0.0.2"
        registry.rollback("cad", "0.0.1")
        assert registry.get("cad").version == "0.0.1"

    def test_list(self, tmp_path):
        source = tmp_path / "src.jsonl"
        source.write_text("line\n", encoding="utf-8")
        registry = DatasetRegistry(tmp_path / "datasets")
        registry.snapshot(name="cad", source=str(source))
        assert registry.list_datasets() == ["cad"]
        assert len(registry.list_versions("cad")) == 1

    def test_missing_source(self, tmp_path):
        registry = DatasetRegistry(tmp_path / "datasets")
        with pytest.raises(FileNotFoundError):
            registry.snapshot(name="cad", source=str(tmp_path / "missing"))

    def test_persistence(self, tmp_path):
        root = tmp_path / "datasets"
        source = tmp_path / "src.jsonl"
        source.write_text("line\n", encoding="utf-8")
        registry = DatasetRegistry(root)
        registry.snapshot(name="cad", source=str(source))
        reloaded = DatasetRegistry(root)
        assert reloaded.get("cad").version == "0.0.1"
