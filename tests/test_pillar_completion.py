from cadgenesis import get_pillar_overview
from cadgenesis.evaluation import BenchmarkSummary, run_pillar_benchmark


def test_pillar_overview_contains_all_major_pillars():
    overview = get_pillar_overview()
    assert isinstance(overview, list)
    assert len(overview) >= 14
    names = {item["name"] for item in overview}
    assert "Pillar 1" in names
    assert "Pillar 14" in names


def test_benchmark_runner_reports_smoke_results():
    report = run_pillar_benchmark()
    assert isinstance(report, BenchmarkSummary)
    assert report.name == "pillar_stack_smoke"
    assert isinstance(report.details, dict)
    assert "transformer" in report.details
