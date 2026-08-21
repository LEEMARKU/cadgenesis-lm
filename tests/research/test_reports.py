from __future__ import annotations

import pytest

from cadgenesis.research.experiments import ExperimentTracker, Hyperparams
from cadgenesis.research.reports import ReportBuilder, experiment_report


class TestReportBuilder:
    def test_markdown(self):
        builder = ReportBuilder(title="Report")
        builder.add_section("Config", {"lr": 1e-4, "notes": ["a", "b"]})
        text = builder.render("markdown")
        assert "# Report" in text
        assert "## Config" in text
        assert "- **lr**: 0.0001" in text

    def test_html(self):
        builder = ReportBuilder(title="Report")
        builder.add_section("Config", {"lr": 1e-4})
        html = builder.render("html")
        assert "<title>Report</title>" in html
        assert "<section>" in html

    def test_pdf_requires_reportlab(self, tmp_path):
        builder = ReportBuilder(title="Report")
        with pytest.raises(RuntimeError):
            builder.render("pdf", path=str(tmp_path / "out.pdf"))

    def test_unknown_format(self):
        builder = ReportBuilder()
        with pytest.raises(ValueError):
            builder.render("docx")

    def test_write_to_file(self, tmp_path):
        builder = ReportBuilder(title="Report")
        builder.add_section("x", {"a": 1})
        out = tmp_path / "report.md"
        builder.render("markdown", path=str(out))
        assert out.exists()
        assert "Report" in out.read_text(encoding="utf-8")

    def test_dashboard_embedded_data(self):
        builder = ReportBuilder(title="Dashboard")
        builder.add_section("x", {"a": 1})
        html = builder.render("dashboard")
        assert "const DATA = {" in html
        assert "filterData" in html


class TestExperimentReport:
    def test_build_from_tracker(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "experiments")
        record = tracker.create(name="run", hyperparams=Hyperparams(learning_rate=1e-4))
        tracker.log_metric(record.id, "loss", 0.3)
        builder = experiment_report(tracker, record.id)
        assert builder.title == f"Experiment {record.id}"

    def test_missing_experiment(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "experiments")
        with pytest.raises(KeyError):
            experiment_report(tracker, "nope")
