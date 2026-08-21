"""
tests/datasets/test_validator.py
================================
Tests for the standalone dataset validator and statistics
(pre-training gate: dataset architecture + validation).
"""

from __future__ import annotations

import pytest

from cadgenesis.datasets.cad_program_synth import build_synthetic_records
from cadgenesis.datasets.validator import (
    DatasetValidator,
    compute_statistics,
    validate_record,
    write_validation_report,
)

GOOD = {
    "text": "a steel box",
    "cad": ["SKETCH_RECT", "EXTRUDE", "BOX", "NUM_80", "NUM_40", "NUM_20"],
    "type": "nl2program",
}


class TestValidateRecord:
    def test_good_record_passes(self):
        checks = validate_record(GOOD)
        assert all(c.passed for c in checks)

    def test_non_dict_record(self):
        checks = validate_record([1, 2, 3])
        assert not checks[0].passed
        assert checks[0].name == "is_dict"

    def test_missing_text(self):
        checks = validate_record({"cad": ["BOX"]})
        assert not _check(checks, "text_present").passed

    def test_empty_cad(self):
        checks = validate_record({"text": "hi", "cad": []})
        assert not _check(checks, "cad_present").passed

    def test_non_string_token(self):
        checks = validate_record({"text": "hi", "cad": ["BOX", 5]})
        assert not _check(checks, "cad_tokens_are_strings").passed

    def test_invalid_token(self):
        checks = validate_record({"text": "hi", "cad": ["BOX", "NOT_A_TOKEN"]})
        # unrecognized UPPER_SNAKE tokens are warnings, not errors
        assert not _check(checks, "cad_tokens_recognized").passed
        assert _check(checks, "cad_tokens_valid").passed

    def test_malformed_token_is_error(self):
        checks = validate_record({"text": "hi", "cad": ["BOX", "not a token!"]})
        assert not _check(checks, "cad_tokens_valid").passed

    def test_numeric_tokens_are_valid(self):
        checks = validate_record({"text": "hi", "cad": ["NUM_0", "NUM_1024"]})
        assert _check(checks, "cad_tokens_valid").passed

    def test_seq_length_limit(self):
        checks = validate_record({"text": "hi", "cad": ["BOX"] * 2000})
        assert not _check(checks, "cad_seq_length").passed

    def test_bad_type_and_metadata(self):
        checks = validate_record({"text": "hi", "cad": ["BOX"], "type": 5, "metadata": []})
        assert not _check(checks, "type_is_string").passed
        assert not _check(checks, "metadata_is_dict").passed

    def test_index_in_detail(self):
        checks = validate_record({"text": "hi", "cad": []}, index=7)
        assert "record 7" in _check(checks, "cad_present").detail


class TestDatasetValidator:
    def test_all_valid(self):
        records = [GOOD, {**GOOD, "text": "a round pin", "cad": ["CYLINDER", "NUM_10", "EXTRUDE", "NUM_40"]}]
        report = DatasetValidator().validate(records)
        assert report.total == 2
        assert report.valid == 2
        assert report.pass_rate == 1.0
        assert report.duplicate_count == 0

    def test_issues_captured(self):
        records = [GOOD, {"text": "broken", "cad": ["bad token!"]}]
        report = DatasetValidator().validate(records)
        assert report.valid == 1
        assert len(report.per_record_issues) == 1
        assert report.per_record_issues[0]["index"] == 1
        assert report.checks_summary["cad_tokens_valid:fail"] == 1

    def test_synthetic_records_pass(self):
        records = build_synthetic_records(50, seed=7)
        report = DatasetValidator().validate(records)
        assert report.pass_rate >= 0.9

    def test_duplicates_detected(self):
        records = [GOOD, dict(GOOD), {**GOOD, "text": "another"}]
        report = DatasetValidator().validate(records)
        assert report.duplicate_count >= 1
        assert report.duplicate_examples


class TestStatistics:
    def test_counts(self):
        records = build_synthetic_records(100, seed=3)
        stats = compute_statistics(records)
        assert stats.total == 100
        assert stats.by_type
        assert stats.total_tokens > 0
        assert stats.avg_cad_len > 0
        assert stats.covered_tokens > 0
        assert 0.0 < stats.vocab_coverage <= 1.0

    def test_top_tokens_sorted(self):
        records = build_synthetic_records(20, seed=5)
        stats = compute_statistics(records)
        frequencies = [count for _, count in stats.token_frequency_top]
        assert frequencies == sorted(frequencies, reverse=True)

    def test_empty_stats(self):
        stats = compute_statistics([])
        assert stats.total == 0
        assert stats.avg_cad_len == 0.0
        assert stats.vocab_coverage == 0.0

    def test_empty_text_and_cad_counted(self):
        records = [{"text": "   ", "cad": []}, {"text": "ok", "cad": ["BOX"]}]
        stats = compute_statistics(records)
        assert stats.empty_text == 1
        assert stats.empty_cad == 1


class TestReport:
    def test_write_report(self, tmp_path):
        records = build_synthetic_records(30, seed=11)
        report = DatasetValidator().validate(records)
        stats = compute_statistics(records)
        path = str(tmp_path / "validation.md")
        text = write_validation_report(path, report, stats, source="synthetic:30")
        assert "Dataset Validation Report" in text
        assert "Pass rate" in text
        assert "Statistics" in text
        assert "Top tokens" in text
        assert text.startswith(open(path, encoding="utf-8").read())

    def test_report_no_statistics(self, tmp_path):
        report = DatasetValidator().validate([GOOD])
        path = str(tmp_path / "simple.md")
        text = write_validation_report(path, report)
        assert "Statistics" not in text


def _check(checks, name):
    return next(c for c in checks if c.name == name)