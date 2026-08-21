from __future__ import annotations

import json

import pytest

from cadgenesis.research.benchmarks import BenchmarkResult, BenchmarkRunner, CADGenerationSuite


class TestBenchmarkRunner:
    def test_builtin_suite(self):
        runner = BenchmarkRunner(seed=42)
        results = runner.run(["cad_generation"])
        assert len(results) == 1
        assert results[0].suite == "cad_generation"
        assert results[0].metrics
        assert results[0].error is None

    def test_all_builtin_suites(self):
        runner = BenchmarkRunner(seed=7)
        results = runner.run()
        assert len(results) >= 6
        assert all(r.error is None for r in results)

    def test_summary_aggregates(self):
        runner = BenchmarkRunner(seed=1)
        results = runner.run(["reasoning"])
        summary = runner.summary(results)
        assert summary["suites_run"] == 1
        assert summary["failures"] == 0
        assert summary["total_duration_s"] >= 0
        assert summary["results"][0]["suite"] == "reasoning"

    def test_save_report(self, tmp_path):
        runner = BenchmarkRunner(seed=1)
        results = runner.run(["planning"])
        path = runner.save_report(results, str(tmp_path / "report.json"))
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        assert payload["results"][0]["suite"] == "planning"

    def test_custom_suite(self):
        runner = BenchmarkRunner(suites=[], seed=1)

        class CustomSuite(CADGenerationSuite):
            @property
            def name(self) -> str:
                return "custom"

            def run(self, seed: int = 42) -> dict:
                return {"score": 0.99}

        runner.register(CustomSuite())
        results = runner.run(["custom"])
        assert results[0].metrics["score"] == 0.99

    def test_unknown_suite(self):
        runner = BenchmarkRunner(seed=1)
        results = runner.run(["nope"])
        assert results[0].error is not None
        assert results[0].metrics == {}

    def test_timed_run_records_duration(self):
        runner = BenchmarkRunner(seed=1)
        result = runner.run(["assembly"])[0]
        assert result.duration_seconds >= 0

    def test_duplicate_registration_rejected(self):
        runner = BenchmarkRunner(suites=[], seed=1)
        with pytest.raises(ValueError):
            runner.register(CADGenerationSuite())


class TestBenchmarkResult:
    def test_to_dict(self):
        result = BenchmarkResult(suite="s", metrics={"a": 1.0}, duration_seconds=0.1, seed=42)
        data = result.to_dict()
        assert data["suite"] == "s"
        assert data["metrics"] == {"a": 1.0}
        assert data["seed"] == 42
