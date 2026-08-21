"""
cadgenesis.datasets.validator
=============================
Standalone dataset validation and statistics for pre-training data.

Provides:

* :func:`validate_record` — per-record schema/quality checks (text/CAD
  presence, token validity against the canonical CAD token registry).
* :class:`DatasetValidator` — batch validation producing a
  :class:`DatasetValidationReport` with per-record issues, duplicate
  detection (MinHash from ``cad_jsonl``) and a dedup report.
* :func:`compute_statistics` — corpus statistics: counts by type, token
  frequency, vocabulary coverage, length distributions.
* :func:`write_validation_report` — markdown report generation.

This is the dataset gate for the pre-training readiness review: loaders
already exist (``cad_jsonl``), this module proves the data is well-formed.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from cadgenesis.datasets.cad_jsonl import minhash_dedup
from cadgenesis.ir import is_base_token, is_feature_token, is_numeric_token
from cadgenesis.tokenizer.cad_tokens import all_cad_token_strings

_SPECIAL_TOKENS = {
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<mask>",
    "<cls>",
    "<sep>",
    "<cad_start>",
    "<cad_end>",
    "<constraint_start>",
    "<constraint_end>",
    "<assembly_start>",
    "<assembly_end>",
    "<material_start>",
    "<material_end>",
    "<manuf_start>",
    "<manuf_end>",
    "<sim_start>",
    "<sim_end>",
    " thinking",
    "<answer>",
    "<memory>",
    "<agent>",
}

MAX_CAD_SEQ_LEN = 1_024


@dataclass(frozen=True)
class RecordCheck:
    """Single check result for one record."""

    name: str
    passed: bool
    severity: str = "error"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "detail": self.detail,
        }


def _valid_token(token: str) -> bool:
    if token in _SPECIAL_TOKENS:
        return True
    if is_numeric_token(token):
        return True
    if is_base_token(token) or is_feature_token(token):
        return True
    return token in _CAD_TOKEN_SET


#: DSL-like shape (``UPPER_SNAKE``): treated as a warning when unrecognized,
#: because the dataset generator's geometry validator accepts a richer DSL
#: vocabulary than the IR legacy keyword set.
_UPPER_SNAKE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


_CAD_TOKEN_SET: frozenset[str] = frozenset(all_cad_token_strings())


def validate_record(record: Any, index: int | None = None) -> list[RecordCheck]:
    """Run schema/quality checks on one dataset record.

    A record is expected to be ``{"text": str, "cad": [str, ...]}`` with the
    optional keys ``type`` (string) and ``metadata`` (dict).
    """
    prefix = f"record {index}" if index is not None else "record"
    checks: list[RecordCheck] = []
    if not isinstance(record, dict):
        return [RecordCheck("is_dict", False, "error", f"{prefix}: not a dict")]

    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        checks.append(
            RecordCheck("text_present", False, "error", f"{prefix}: missing non-empty 'text'")
        )
    else:
        checks.append(RecordCheck("text_present", True, "error", ""))

    cad = record.get("cad")
    if not isinstance(cad, list) or not cad:
        checks.append(
            RecordCheck("cad_present", False, "error", f"{prefix}: missing non-empty 'cad' list")
        )
    elif not all(isinstance(t, str) for t in cad):
        checks.append(
            RecordCheck("cad_tokens_are_strings", False, "error", f"{prefix}: non-string token")
        )
    else:
        checks.append(RecordCheck("cad_tokens_are_strings", True, "error", ""))
        if len(cad) > MAX_CAD_SEQ_LEN:
            checks.append(
                RecordCheck(
                    "cad_seq_length",
                    False,
                    "error",
                    f"{prefix}: cad length {len(cad)} exceeds {MAX_CAD_SEQ_LEN}",
                )
            )
        else:
            checks.append(RecordCheck("cad_seq_length", True, "error", ""))
        invalid = [t for t in cad if not isinstance(t, str) or not _valid_token(t)]
        malformed = [t for t in invalid if not (isinstance(t, str) and _UPPER_SNAKE_RE.match(t))]
        unrecognized = [t for t in invalid if isinstance(t, str) and _UPPER_SNAKE_RE.match(t)]
        if malformed:
            sample = ", ".join(sorted(set(map(str, malformed)))[:5])
            checks.append(
                RecordCheck(
                    "cad_tokens_valid",
                    False,
                    "error",
                    f"{prefix}: malformed tokens: {sample}",
                )
            )
        else:
            checks.append(RecordCheck("cad_tokens_valid", True, "error", ""))
        if unrecognized:
            sample = ", ".join(sorted(set(unrecognized))[:5])
            checks.append(
                RecordCheck(
                    "cad_tokens_recognized",
                    False,
                    "warning",
                    f"{prefix}: unrecognized DSL tokens (accepted by geometry validator): {sample}",
                )
            )
        else:
            checks.append(RecordCheck("cad_tokens_recognized", True, "warning", ""))

    record_type = record.get("type")
    if record_type is not None and not isinstance(record_type, str):
        checks.append(
            RecordCheck("type_is_string", False, "error", f"{prefix}: 'type' must be a string")
        )
    else:
        checks.append(RecordCheck("type_is_string", True, "error", ""))

    metadata = record.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        checks.append(
            RecordCheck(
                "metadata_is_dict", False, "error", f"{prefix}: 'metadata' must be a dict"
            )
        )
    else:
        checks.append(RecordCheck("metadata_is_dict", True, "error", ""))

    return checks


def _record_passed(checks: list[RecordCheck]) -> bool:
    return all(c.passed for c in checks if c.severity == "error")


@dataclass
class DatasetValidationReport:
    """Aggregated validation outcome over a corpus."""

    total: int
    valid: int
    per_record_issues: list[dict[str, Any]] = field(default_factory=list)
    duplicate_count: int = 0
    duplicate_examples: list[str] = field(default_factory=list)
    checks_summary: dict[str, int] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.valid / self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "valid": self.valid,
            "pass_rate": round(self.pass_rate, 4),
            "duplicate_count": self.duplicate_count,
            "duplicate_examples": self.duplicate_examples,
            "checks_summary": dict(self.checks_summary),
            "per_record_issues": list(self.per_record_issues),
        }


class DatasetValidator:
    """Batch-validate records and report duplicates + check summary."""

    def __init__(self, allowed_tokens: set[str] | None = None):
        self.allowed_tokens = allowed_tokens

    def validate(self, records: list[dict[str, Any]]) -> DatasetValidationReport:
        report = DatasetValidationReport(total=len(records), valid=0)
        check_counts: Counter[str] = Counter()
        valid = 0
        for index, record in enumerate(records):
            checks = validate_record(record, index)
            for check in checks:
                check_counts[f"{check.name}:{'pass' if check.passed else 'fail'}"] += 1
            if _record_passed(checks):
                valid += 1
            else:
                report.per_record_issues.append(
                    {
                        "index": index,
                        "checks": [c.to_dict() for c in checks if not c.passed],
                    }
                )
        report.valid = valid
        report.checks_summary = {
            name: count for name, count in sorted(check_counts.items())
        }
        if records:
            kept = minhash_dedup(records, progress=False)
            report.duplicate_count = len(records) - len(kept)
            kept_texts = {r.get("text", "") for r in kept}
            report.duplicate_examples = [
                str(r.get("text", ""))[:80]
                for r in records
                if str(r.get("text", "")) not in kept_texts
            ][:5]
        return report


@dataclass
class DatasetStatistics:
    """Corpus statistics for the dataset readiness gate."""

    total: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    token_frequency_top: list[tuple[str, int]] = field(default_factory=list)
    vocab_coverage: float = 0.0
    covered_tokens: int = 0
    total_vocab_tokens: int = 0
    avg_text_len: float = 0.0
    avg_cad_len: float = 0.0
    empty_text: int = 0
    empty_cad: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "by_type": dict(self.by_type),
            "token_frequency_top": list(self.token_frequency_top),
            "vocab_coverage": round(self.vocab_coverage, 4),
            "covered_tokens": self.covered_tokens,
            "total_vocab_tokens": self.total_vocab_tokens,
            "avg_text_len": round(self.avg_text_len, 2),
            "avg_cad_len": round(self.avg_cad_len, 2),
            "empty_text": self.empty_text,
            "empty_cad": self.empty_cad,
            "total_tokens": self.total_tokens,
        }


def compute_statistics(records: list[dict[str, Any]], top_n: int = 20) -> DatasetStatistics:
    """Compute corpus statistics over validated records."""
    stats = DatasetStatistics(total=len(records))
    token_counter: Counter[str] = Counter()
    text_lengths: list[int] = []
    cad_lengths: list[int] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        record_type = record.get("type", "other")
        stats.by_type[record_type] = stats.by_type.get(record_type, 0) + 1
        text = record.get("text", "")
        if isinstance(text, str):
            if not text.strip():
                stats.empty_text += 1
            text_lengths.append(len(text.split()))
        cad = record.get("cad", [])
        if isinstance(cad, list) and cad:
            cad_lengths.append(len(cad))
            token_counter.update(t for t in cad if isinstance(t, str))
        elif isinstance(cad, list):
            stats.empty_cad += 1
    stats.token_frequency_top = token_counter.most_common(top_n)
    stats.total_tokens = sum(token_counter.values())
    stats.covered_tokens = len(set(token_counter))
    stats.total_vocab_tokens = len(_CAD_TOKEN_SET)
    stats.vocab_coverage = (
        stats.covered_tokens / stats.total_vocab_tokens if stats.total_vocab_tokens else 0.0
    )
    stats.avg_text_len = sum(text_lengths) / len(text_lengths) if text_lengths else 0.0
    stats.avg_cad_len = sum(cad_lengths) / len(cad_lengths) if cad_lengths else 0.0
    return stats


def write_validation_report(
    path: str,
    report: DatasetValidationReport,
    statistics: DatasetStatistics | None = None,
    source: str = "",
) -> str:
    """Write a markdown dataset-validation report; returns the text."""
    lines = ["# Dataset Validation Report", ""]
    if source:
        lines += [f"**Source:** `{source}`", ""]
    lines += [
        f"**Total records:** {report.total}",
        f"**Valid records:** {report.valid}",
        f"**Pass rate:** {report.pass_rate:.2%}",
        f"**Duplicate records (MinHash):** {report.duplicate_count}",
        "",
        "## Check Summary",
        "",
    ]
    if report.checks_summary:
        lines += [f"- {name}: {count}" for name, count in report.checks_summary.items()]
    else:
        lines += ["- (no checks recorded)"]
    lines.append("")
    if report.duplicate_examples:
        lines += ["## Duplicate Examples", ""]
        lines += [f"- {text}" for text in report.duplicate_examples]
        lines.append("")
    if statistics is not None:
        lines += [
            "## Statistics",
            "",
            f"- Records: {statistics.total}",
            f"- Tokens total: {statistics.total_tokens}",
            f"- Vocabulary coverage: {statistics.vocab_coverage:.2%} "
            f"({statistics.covered_tokens}/{statistics.total_vocab_tokens})",
            f"- Avg text length (words): {statistics.avg_text_len:.2f}",
            f"- Avg CAD length (tokens): {statistics.avg_cad_len:.2f}",
            "",
            "### Records by type",
            "",
        ]
        if statistics.by_type:
            lines += [
                f"- {name}: {count}" for name, count in sorted(statistics.by_type.items())
            ]
        lines += ["", "### Top tokens", ""]
        if statistics.token_frequency_top:
            lines += [
                f"- {token}: {count}" for token, count in statistics.token_frequency_top
            ]
        lines.append("")
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return text


__all__ = [
    "DatasetStatistics",
    "DatasetValidationReport",
    "DatasetValidator",
    "RecordCheck",
    "compute_statistics",
    "validate_record",
    "write_validation_report",
]