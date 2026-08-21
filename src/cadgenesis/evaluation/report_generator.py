"""cadgenesis.evaluation.report_generator
======================================
Evaluation report generation.

Pure string building: markdown documents with bullet-list metrics and
tables from ``list[dict]`` rows, indented JSON, and a top-level convenience
``generate_report`` entry point.
"""

from __future__ import annotations

import json
from typing import Any

_VALID_FORMATS = ("markdown", "json")


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


class ReportGenerator:
    """Render evaluation metrics into markdown or JSON reports."""

    def render_markdown(
        self,
        sections: dict[str, Any],
        title: str = "Evaluation Report",
    ) -> str:
        """Render sections into a markdown document.

        A section whose content is a ``dict`` becomes a bullet list of
        ``key: value`` lines; a ``list[dict]`` becomes a table; any other
        list becomes bullet lines.
        """
        lines: list[str] = [f"# {title}", ""]
        for name, content in sections.items():
            lines.append(f"## {name}")
            lines.append("")
            if isinstance(content, dict):
                lines.extend(f"- {key}: {_format_value(value)}" for key, value in content.items())
            elif isinstance(content, list):
                if content and all(isinstance(row, dict) for row in content):
                    lines.append(self.to_markdown_table(content))
                else:
                    lines.extend(f"- {_format_value(item)}" for item in content)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def render_json(report: dict[str, Any]) -> str:
        """Serialize the report dict as indented JSON."""
        return json.dumps(report, indent=2)

    @staticmethod
    def to_markdown_table(rows: list[dict[str, Any]]) -> str:
        """Render ``list[dict]`` rows as a markdown table.

        Columns are all keys in first-seen order across rows.  An empty
        row list yields an empty string.
        """
        if not rows:
            return ""
        headers: list[str] = []
        for row in rows:
            for key in row:
                if key not in headers:
                    headers.append(str(key))
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("|" + "|".join([" --- "] * len(headers)) + "|")
        lines.extend(
            "| " + " | ".join(_format_value(row.get(h)) for h in headers) + " |" for row in rows
        )
        return "\n".join(lines)


def generate_report(
    metrics: dict[str, dict[str, Any]],
    output_format: str = "markdown",
) -> str:
    """Convenience wrapper rendering ``metrics`` in the requested format."""
    if output_format not in _VALID_FORMATS:
        raise ValueError(
            f"Unsupported output format: {output_format!r} "
            f"(expected one of {', '.join(_VALID_FORMATS)})"
        )
    generator = ReportGenerator()
    if output_format == "json":
        return generator.render_json(metrics)
    return generator.render_markdown(metrics)


__all__ = ["ReportGenerator", "generate_report"]
