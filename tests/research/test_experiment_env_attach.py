"""
tests/research/test_experiment_env_attach.py
============================================
Tests for experiment reproducibility metadata (pre-training gate:
every experiment must record its environment and dataset versions).
"""

from __future__ import annotations

import json
from pathlib import Path

from cadgenesis.research.integration import ResearchSession


class TestExperimentEnvAttach:
    def test_environment_snapshot_attached(self, tmp_path: Path):
        session = ResearchSession(str(tmp_path))
        experiment = session.new_experiment(
            name="env-test",
            hyperparams={"learning_rate": 1e-4, "batch_size": 8},
        )
        env = experiment.metadata.get("environment", {})
        assert isinstance(env, dict)
        assert env.get("python_version")
        assert env.get("platform") or env.get("os")

    def test_dataset_versions_attached(self, tmp_path: Path):
        session = ResearchSession(str(tmp_path))
        session.snapshot_dataset(
            source=__file__,
            name="bench_manual",
            metadata={"note": "held-out"},
        )
        experiment = session.new_experiment(name="dataset-test")
        datasets = experiment.metadata.get("datasets", [])
        assert any(d["name"] == "bench_manual" for d in datasets)
        assert datasets[0]["versions"]

    def test_no_datasets_when_registry_empty(self, tmp_path: Path):
        session = ResearchSession(str(tmp_path))
        experiment = session.new_experiment(name="clean")
        assert experiment.metadata.get("datasets") == []

    def test_opt_out(self, tmp_path: Path):
        session = ResearchSession(str(tmp_path))
        experiment = session.new_experiment(name="minimal", track_environment=False, track_datasets=False)
        assert "environment" not in experiment.metadata
        assert "datasets" not in experiment.metadata

    def test_record_persisted_with_env(self, tmp_path: Path):
        session = ResearchSession(str(tmp_path))
        experiment = session.new_experiment(name="persist-env")
        reloaded = session.tracker.get(experiment.id)
        assert reloaded is not None
        assert reloaded.metadata.get("environment", {}).get("python_version")

    def test_environment_to_dict_serializable(self, tmp_path: Path):
        session = ResearchSession(str(tmp_path))
        env = session.environment()
        json.dumps(env)  # must not raise