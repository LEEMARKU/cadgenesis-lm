"""cadgenesis.execution.feedback
==============================
Execution → model feedback loop.

Collects findings from geometry/topology/manufacturing/simulation/
optimization/repair reports and folds them into a
:class:`CADExecutionResult` as errors and suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FeedbackItem:
    """One actionable feedback entry."""

    source: str
    message: str
    severity: str = "info"
    suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "message": self.message,
            "severity": self.severity,
            "suggestion": self.suggestion,
        }


class FeedbackLoop:
    """Aggregates report findings into execution feedback."""

    def collect(self, reports: dict[str, Any]) -> list[FeedbackItem]:
        """Collect feedback items from a dict of named reports.

        Accepted report shapes: objects with ``checks`` (each with ``passed``,
        ``name``, ``detail``, ``recommendation``/``suggestion``), or plain
        dicts with ``summary()``.
        """
        items: list[FeedbackItem] = []
        for source, report in reports.items():
            if report is None:
                continue
            if isinstance(report, dict):
                failed = report.get("failed")
                if failed:
                    items.append(
                        FeedbackItem(
                            source=source,
                            message=f"failed checks: {', '.join(map(str, failed))}",
                            severity="error",
                        )
                    )
                items.extend(
                    FeedbackItem(
                        source=source,
                        message=str(suggestion),
                        severity="info",
                        suggestion=str(suggestion),
                    )
                    for suggestion in report.get("suggestions") or []
                )
                continue
            checks = getattr(report, "checks", None)
            if isinstance(checks, list):
                for check in checks:
                    if getattr(check, "passed", True):
                        continue
                    recommendation = getattr(check, "recommendation", "") or getattr(
                        check, "suggestion", ""
                    )
                    items.append(
                        FeedbackItem(
                            source=source,
                            message=str(getattr(check, "detail", "") or getattr(check, "name", "")),
                            severity=str(getattr(check, "severity", "error")),
                            suggestion=recommendation,
                        )
                    )
                continue
            summary = getattr(report, "summary", None)
            if callable(summary):
                data = summary()
                failed = data.get("failed") if isinstance(data, dict) else None
                if failed:
                    items.append(
                        FeedbackItem(
                            source=source,
                            message=f"failed checks: {', '.join(map(str, failed))}",
                            severity="error",
                        )
                    )
        return items

    def apply(
        self,
        result: Any,
        reports: dict[str, Any],
    ) -> Any:
        """Fold collected feedback into a result-like object.

        ``result`` needs mutable ``errors``/``suggestions`` lists (the
        :class:`CADExecutionResult` contract); returns it unchanged.
        """
        for item in self.collect(reports):
            if item.severity == "error":
                result.errors.append(f"[{item.source}] {item.message}")
            elif item.suggestion:
                result.suggestions.append(f"[{item.source}] {item.suggestion}")
            else:
                result.suggestions.append(f"[{item.source}] {item.message}")
        return result

    def to_dict(self, reports: dict[str, Any]) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.collect(reports)]

    def feedback_on_diff(self, diff_report: Any) -> list[FeedbackItem]:
        """Feedback items from an IR diff report (v6.4 feedback loop).

        Every added/removed/changed operation becomes an informational item;
        structural changes (ops removed or changed) are reported at
        ``warning`` severity so revision feedback is visible downstream.
        """
        items = [
            FeedbackItem(
                source="ir-diff",
                message=f"added {op.get('kind')} @ position {op.get('position')}",
                severity="info",
            )
            for op in getattr(diff_report, "added", [])
        ]
        items.extend(
            FeedbackItem(
                source="ir-diff",
                message=f"removed {op.get('kind')} @ position {op.get('position')}",
                severity="warning",
            )
            for op in getattr(diff_report, "removed", [])
        )
        items.extend(
            FeedbackItem(
                source="ir-diff",
                message=(
                    f"changed {op.get('kind')} @ position {op.get('position')}: "
                    f"{', '.join(op.get('changed_params') or [])}"
                ),
                severity="warning",
            )
            for op in getattr(diff_report, "changed", [])
        )
        return items


__all__ = ["FeedbackItem", "FeedbackLoop"]
