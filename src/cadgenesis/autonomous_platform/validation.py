"""
End-to-End Validation - CAD correctness, engineering correctness, manufacturability, simulation
quality, documentation, safety, explainability.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any


class ValidationCategory(str, Enum):
    CAD_CORRECTNESS = "cad_correctness"
    ENGINEERING_CORRECTNESS = "engineering_correctness"
    MANUFACTURABILITY = "manufacturability"
    SIMULATION_QUALITY = "simulation_quality"
    DOCUMENTATION = "documentation"
    SAFETY = "safety"
    EXPLAINABILITY = "explainability"


class ValidationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    check_id: str
    category: ValidationCategory
    name: str
    status: ValidationStatus
    score: float  # 0-1
    details: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ValidationReport:
    """Complete validation report."""

    report_id: str
    workflow_id: str
    results: list[ValidationResult] = field(default_factory=list)
    overall_score: float = 0.0
    overall_status: ValidationStatus = ValidationStatus.PASSED
    summary: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


ValidatorFn = Callable[[Any], ValidationResult]


class EndToEndValidator:
    """Validates all aspects of the generated engineering solution."""

    def __init__(self):
        self._validators: dict[ValidationCategory, list[tuple[str, ValidatorFn]]] = {}
        self._thresholds: dict[str, float] = {}
        self._reports: dict[str, ValidationReport] = {}
        self._lock = RLock()

    def register_validator(
        self,
        category: ValidationCategory,
        name: str,
        function: ValidatorFn,
        threshold: float = 0.8,
    ) -> None:
        """Register a validation function."""
        with self._lock:
            if category not in self._validators:
                self._validators[category] = []
            self._validators[category].append((name, function))
            self._thresholds[name] = threshold

    def validate(
        self,
        workflow_id: str,
        generated_artifact: Any,
        context: dict[str, Any],
    ) -> ValidationReport:
        """Run all validations on the generated artifact."""
        report = ValidationReport(
            report_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
        )

        all_passed = True
        total_score = 0.0
        count = 0

        for category, validators in self._validators.items():
            for name, validator in validators:
                try:
                    result = validator(generated_artifact)

                    # Apply threshold
                    if result.score < self._thresholds.get(name, 0.8):
                        result.status = ValidationStatus.FAILED
                        all_passed = False
                    elif result.status == ValidationStatus.PASSED:
                        pass  # Keep as passed

                    report.results.append(result)
                    total_score += result.score
                    count += 1

                except Exception as e:
                    result = ValidationResult(
                        check_id=str(uuid.uuid4()),
                        category=category,
                        name=name,
                        status=ValidationStatus.FAILED,
                        score=0.0,
                        details=f"Validation error: {e}",
                    )
                    report.results.append(result)
                    all_passed = False
                    count += 1

        report.overall_score = total_score / count if count > 0 else 0.0
        report.overall_status = ValidationStatus.PASSED if all_passed else ValidationStatus.FAILED

        # Generate summary
        passed = sum(1 for r in report.results if r.status == ValidationStatus.PASSED)
        failed = sum(1 for r in report.results if r.status == ValidationStatus.FAILED)
        warning = sum(1 for r in report.results if r.status == ValidationStatus.WARNING)
        report.summary = (
            f"Validation: {passed} passed, {failed} failed, {warning} warnings. "
            f"Overall score: {report.overall_score:.3f}"
        )

        with self._lock:
            self._reports[report.report_id] = report

        return report

    def get_report(self, report_id: str) -> ValidationReport | None:
        with self._lock:
            return self._reports.get(report_id)

    def list_reports(self, workflow_id: str | None = None) -> list[ValidationReport]:
        with self._lock:
            reports = list(self._reports.values())
            if workflow_id:
                reports = [r for r in reports if r.workflow_id == workflow_id]
            return reports
