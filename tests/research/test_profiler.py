from __future__ import annotations

import time

from cadgenesis.research.profiler import PerformanceProfiler, SystemSnapshot


class TestSystemSnapshot:
    def test_to_dict(self):
        snapshot = SystemSnapshot(timestamp=1.0, cpu_percent=10.0, memory_percent=50.0)
        data = snapshot.to_dict()
        assert data["cpu_percent"] == 10.0
        assert data["gpu_util"] == 0.0


class TestPerformanceProfiler:
    def test_sampler_collects(self):
        profiler = PerformanceProfiler(sample_interval=0.1)
        profiler.start()
        time.sleep(0.35)
        profiler.stop()
        assert len(profiler.snapshots) >= 1

    def test_disabled_profiler(self):
        profiler = PerformanceProfiler(sample_interval=0.1, enabled=False)
        profiler.start()
        time.sleep(0.2)
        profiler.stop()
        assert profiler.snapshots == []

    def test_profile_inference(self):
        profiler = PerformanceProfiler(enabled=False)
        results = profiler.profile_inference(lambda: None, batch_sizes=[1, 2], repeats=2)
        assert "batch_1_ms" in results
        assert "batch_2_ms" in results
        assert results["batch_1_ms"] >= 0

    def test_time_phase(self):
        profiler = PerformanceProfiler(enabled=False)
        profiler.time_phase("forward", lambda: time.sleep(0.001))
        profiler.time_phase("forward", lambda: time.sleep(0.001))
        assert profiler.phases["forward"] > 0.002

    def test_summary_without_snapshots(self):
        profiler = PerformanceProfiler(enabled=False)
        summary = profiler.summary()
        assert summary == {"phases": {}}

    def test_summary_with_snapshots(self):
        profiler = PerformanceProfiler(sample_interval=0.1)
        profiler.start()
        time.sleep(0.35)
        profiler.stop()
        summary = profiler.summary()
        assert summary["samples"] >= 1
        assert "avg_cpu_percent" in summary
