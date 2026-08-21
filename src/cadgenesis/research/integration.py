"""
cadgenesis.research.integration
===============================
Research infrastructure integration layer.

Glues experiments, benchmarks, datasets, ablation, reproducibility,
artifacts and reports into a single launch-and-track workflow:

- ``ResearchSession``: one root directory, one tracker, one artifact
  registry, and helpers to snapshot datasets, run benchmarks and
  experiments with deterministic seeds
- ``launch_experiment``: script entry point that wires hyperparameters,
  environment capture and report generation around a user runnable
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from cadgenesis.research.artifacts import ArtifactRegistry
from cadgenesis.research.experiments import ExperimentTracker
from cadgenesis.research.reports import ReportBuilder
from cadgenesis.research.reproducibility import EnvironmentCapture, SeedRegistry

logger = logging.getLogger("cadgenesis.research.integration")


class ResearchSession:
    """Top-level integration: tracker + artifacts + seeds in one root."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.tracker = ExperimentTracker(self.root / "experiments")
        self.artifacts = ArtifactRegistry(self.root / "artifacts")
        self.seeds = SeedRegistry(42)
        self.reports: list[str] = []

    def snapshot_dataset(
        self,
        source: str | os.PathLike[str],
        name: str,
        version: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Register a dataset snapshot under the session root."""
        from cadgenesis.research.datasets import DatasetRegistry

        registry = DatasetRegistry(self.root / "datasets")
        return registry.snapshot(name=name, source=source, version=version, metadata=metadata)

    def new_experiment(
        self,
        name: str,
        hyperparams: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        track_environment: bool = True,
        track_datasets: bool = True,
    ) -> Any:
        """Create and return a fresh experiment record.

        When ``track_environment`` is set, a full environment snapshot
        (python/torch/CUDA/platform/hostname) is embedded in the record's
        ``environment`` metadata.  When ``track_datasets`` is set, the
        registered dataset versions from the session's dataset registry are
        embedded under ``datasets`` — either feature makes a run reproducible.
        """
        from cadgenesis.research.experiments import Hyperparams

        hp = Hyperparams(**dict(hyperparams or {}))
        extra = dict(metadata or {})
        if track_environment:
            extra["environment"] = self.environment()
        if track_datasets:
            registry_root = self.root / "datasets"
            if registry_root.exists():
                from cadgenesis.research.datasets import DatasetRegistry

                registry = DatasetRegistry(registry_root)
                extra["datasets"] = [
                    {"name": d, "versions": [v.to_dict() for v in registry.list_versions(d)]}
                    for d in registry.list_datasets()
                ]
            else:
                extra["datasets"] = []
        return self.tracker.create(name=name, hyperparams=hp, metadata=extra)

    def run_benchmark(self, suite: str, seed: int | None = None) -> Any:
        """Run a builtin benchmark suite and store its report."""
        from cadgenesis.research.benchmarks import BenchmarkRunner

        runner = BenchmarkRunner(seed=seed or self.seeds.seed_for(suite))
        results = runner.run([suite])
        summary = runner.summary(results)
        builder = ReportBuilder(title=f"Benchmark suite {suite}")
        builder.add_section("Results", [r.to_dict() for r in results])
        report = builder.render("markdown")
        self.reports.append(report)
        return summary

    def environment(self) -> Mapping[str, Any]:
        """Current environment capture for reproducibility."""
        return EnvironmentCapture.capture().to_dict()


def launch_experiment(
    run_fn: Callable[..., Mapping[str, float]],
    root: str,
    name: str,
    hyperparams: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    seed: int | None = None,
    metric: str = "loss",
    minimize: bool = True,
    report_path: str | None = None,
) -> dict[str, Any]:
    """Convenience launcher: track one run end-to-end and return its record.

    ``run_fn(hyperparams: dict, seed: int) -> dict[str, float]`` receives
    plain hyperparameters and a deterministic seed; returned metrics are
    logged on the experiment.
    """
    session = ResearchSession(root)
    experiment = session.new_experiment(name=name, hyperparams=hyperparams, metadata=metadata)
    effective_seed = seed if seed is not None else session.seeds.seed_for(name)
    try:
        metrics = run_fn(dict(hyperparams or {}), effective_seed)
        for key, value in metrics.items():
            session.tracker.log_metric(experiment.id, key, float(value))
        session.tracker.set_status(experiment.id, "completed", metric=metric, minimize=minimize)
    except Exception as exc:
        session.tracker.add_note(experiment.id, f"failed: {type(exc).__name__}: {exc}")
        session.tracker.set_status(experiment.id, "failed", metric=metric, minimize=minimize)
        raise
    record = session.tracker.get(experiment.id)
    assert record is not None
    if report_path:
        builder = ReportBuilder(title=f"Experiment {name}")
        builder.add_section("Record", record.to_dict())
        builder.add_section("Environment", session.environment())
        builder.render("markdown", path=report_path)
    return {
        "experiment": experiment.id,
        "seed": effective_seed,
        "record": record.to_dict(),
    }


__all__ = ["ResearchSession", "launch_experiment"]
