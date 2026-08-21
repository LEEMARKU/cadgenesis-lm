"""tests/transformer/evolution/test_experiments.py
=====================================================
Unit tests for the experiment registry.
"""

from __future__ import annotations

import pytest

from cadgenesis.transformer.evolution.experiments import (
    ExperimentRecord,
    ExperimentRegistry,
)


class TestExperimentRegistry:
    def test_log_and_get(self):
        reg = ExperimentRegistry()
        rec = reg.log(
            "run-1",
            arch_version="hier@1.0.0",
            config_hash="abcd",
            metric=0.92,
            metadata={"lr": 1e-4},
        )
        assert isinstance(rec, ExperimentRecord)
        assert reg.get("run-1").metric == 0.92

    def test_best(self):
        reg = ExperimentRegistry()
        reg.log("a", arch_version="v", config_hash="h", metric=0.5)
        reg.log("b", arch_version="v", config_hash="h", metric=0.9)
        assert reg.best().name == "b"
        assert reg.best(higher_is_better=False).name == "a"

    def test_by_arch_version(self):
        reg = ExperimentRegistry()
        reg.log("a", arch_version="v1", config_hash="h", metric=1.0)
        reg.log("b", arch_version="v2", config_hash="h", metric=2.0)
        assert [r.name for r in reg.by_arch_version("v2")] == ["b"]

    def test_persistence(self, tmp_path):
        path = tmp_path / "experiments.json"
        reg = ExperimentRegistry()
        reg.log("x", arch_version="v", config_hash="h", metric=3.0)
        reg.save(str(path))
        loaded = ExperimentRegistry(str(path))
        assert loaded.get("x").metric == 3.0
        assert loaded.summary()["count"] == 1

    def test_summary(self):
        reg = ExperimentRegistry()
        assert reg.best() is None
        reg.log("a", arch_version="v", config_hash="h", metric=0.1)
        s = reg.summary()
        assert s["count"] == 1
        assert s["best"]["name"] == "a"

    def test_empty_name_rejected(self):
        reg = ExperimentRegistry()
        with pytest.raises(ValueError):
            reg.log("", arch_version="v", config_hash="h", metric=1.0)

    def test_get_unknown(self):
        with pytest.raises(KeyError):
            ExperimentRegistry().get("missing")
