"""tests/evaluation/test_report_generator.py
============================================
Unit tests for evaluation report generation.
"""

from __future__ import annotations

import json

import pytest

from cadgenesis.evaluation.report_generator import ReportGenerator, generate_report


def test_to_markdown_table():
    rows = [{"metric": "validity", "score": 1.0}, {"metric": "fidelity", "score": 0.75}]
    table = ReportGenerator.to_markdown_table(rows)
    assert "| metric | score |" in table
    assert "| --- | --- |" in table
    assert "| validity | 1.0000 |" in table
    assert "| fidelity | 0.7500 |" in table
    assert ReportGenerator.to_markdown_table([]) == ""


def test_to_markdown_table_heterogeneous_keys():
    rows = [{"metric": "a", "score": 1.0}, {"metric": "b", "score": 0.5, "note": "ok"}]
    table = ReportGenerator.to_markdown_table(rows)
    assert "| metric | score | note |" in table
    assert "| b | 0.5000 | ok |" in table


def test_render_markdown():
    report = ReportGenerator().render_markdown(
        {"validity": {"rate": 1.0}, "per_sample": [{"metric": "acc", "value": 0.5}]},
        title="My Report",
    )
    assert report.startswith("# My Report")
    assert "## validity" in report
    assert "- rate: 1.0000" in report
    assert "## per_sample" in report
    assert "| metric | value |" in report
    assert "| --- | --- |" in report
    assert "| acc | 0.5000 |" in report


def test_render_markdown_default_title():
    report = ReportGenerator().render_markdown({})
    assert report == "# Evaluation Report\n"


def test_render_json():
    text = ReportGenerator.render_json({"a": {"b": 1.0}})
    assert json.loads(text) == {"a": {"b": 1.0}}
    assert text.startswith('{\n  "a": {')


def test_generate_report():
    metrics = {"validity": {"rate": 1.0}, "counts": {"samples": 5}}
    md = generate_report(metrics)
    assert md.startswith("# Evaluation Report")
    assert "- rate: 1.0000" in md
    js = generate_report(metrics, output_format="json")
    assert json.loads(js) == metrics
    with pytest.raises(ValueError, match="output format"):
        generate_report(metrics, output_format="xml")
