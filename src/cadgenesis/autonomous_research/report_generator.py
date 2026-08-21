"""Research Report Generator - PDF, Markdown, HTML, Interactive Dashboard."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any


class ReportFormat(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    DASHBOARD = "dashboard"


@dataclass
class ReportSection:
    """A section in the research report."""

    title: str
    content: str
    level: int = 1  # 1=h1, 2=h2, etc.
    figures: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ResearchReport:
    """A complete research report."""

    report_id: str
    title: str
    authors: list[str]
    abstract: str
    sections: list[ReportSection] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    figures: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    conclusions: str = ""
    references: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class ResearchReportGenerator:
    """Generates research reports in multiple formats."""

    def __init__(self, output_dir: str = "./research_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._reports: dict[str, ResearchReport] = {}
        self._lock = RLock()

    def create_report(
        self,
        title: str,
        authors: list[str],
        abstract: str,
    ) -> ResearchReport:
        report = ResearchReport(
            report_id=str(uuid.uuid4()),
            title=title,
            authors=authors,
            abstract=abstract,
        )
        with self._lock:
            self._reports[report.report_id] = report
        return report

    def add_section(
        self,
        report_id: str,
        title: str,
        content: str,
        level: int = 1,
        figures: list[dict] | None = None,
        tables: list[dict] | None = None,
    ) -> bool:
        with self._lock:
            report = self._reports.get(report_id)
            if not report:
                return False
            section = ReportSection(
                title=title,
                content=content,
                level=level,
                figures=figures or [],
                tables=tables or [],
            )
            report.sections.append(section)
            return True

    def add_metrics(self, report_id: str, metrics: dict[str, float]) -> bool:
        with self._lock:
            report = self._reports.get(report_id)
            if not report:
                return False
            report.metrics.update(metrics)
            return True

    def add_conclusions(self, report_id: str, conclusions: str) -> bool:
        with self._lock:
            report = self._reports.get(report_id)
            if not report:
                return False
            report.conclusions = conclusions
            return True

    def generate(self, report_id: str, format: ReportFormat) -> str:
        """Generate report in specified format."""
        with self._lock:
            report = self._reports.get(report_id)
            if not report:
                raise ValueError(f"Report {report_id} not found")

        if format == ReportFormat.MARKDOWN:
            return self._generate_markdown(report)
        elif format == ReportFormat.HTML:
            return self._generate_html(report)
        elif format == ReportFormat.DASHBOARD:
            return self._generate_dashboard(report)
        elif format == ReportFormat.PDF:
            return self._generate_pdf(report)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_markdown(self, report: ResearchReport) -> str:
        lines = [
            f"# {report.title}",
            "",
            f"**Authors:** {', '.join(report.authors)}",
            f"**Date:** {time.strftime('%Y-%m-%d', time.localtime(report.created_at))}",
            f"**Report ID:** {report.report_id}",
            "",
            "## Abstract",
            report.abstract,
            "",
        ]

        for section in report.sections:
            lines.append(f"{'#' * (section.level + 1)} {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")

            for fig in section.figures:
                lines.append(f"![{fig.get('caption', 'Figure')}]({fig.get('path', '')})")
                lines.append("")

            for table in section.tables:
                lines.append(self._format_table(table))
                lines.append("")

        if report.metrics:
            lines.append("## Metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in report.metrics.items():
                lines.append(f"| {k} | {v:.4f} |")
            lines.append("")

        if report.conclusions:
            lines.append("## Conclusions")
            lines.append("")
            lines.append(report.conclusions)
            lines.append("")

        if report.references:
            lines.append("## References")
            lines.append("")
            for ref in report.references:
                lines.append(f"- {ref}")

        content = "\n".join(lines)

        # Save to file
        path = self.output_dir / f"{report.report_id}.md"
        with open(path, "w") as f:
            f.write(content)

        return content

    def _generate_html(self, report: ResearchReport) -> str:
        md = self._generate_markdown(report)
        # Simple markdown to HTML conversion
        html = md.replace("# ", "<h1>").replace("\n# ", "</h1>\n<h1>")
        html = html.replace("## ", "<h2>").replace("\n## ", "</h2>\n<h2>")
        html = html.replace("### ", "<h3>").replace("\n### ", "</h3>\n<h3>")
        html = html.replace("\n\n", "</p>\n<p>")
        html = f"<html><body><p>{html}</p></body></html>"

        path = self.output_dir / f"{report.report_id}.html"
        with open(path, "w") as f:
            f.write(html)

        return html

    def _generate_dashboard(self, report: ResearchReport) -> str:
        """Generate interactive dashboard data (JSON for frontend)."""
        dashboard_data = {
            "report_id": report.report_id,
            "title": report.title,
            "authors": report.authors,
            "abstract": report.abstract,
            "metrics": report.metrics,
            "sections": [
                {
                    "title": s.title,
                    "content": s.content,
                    "level": s.level,
                    "figures": s.figures,
                    "tables": s.tables,
                }
                for s in report.sections
            ],
            "conclusions": report.conclusions,
            "references": report.references,
        }

        path = self.output_dir / f"{report.report_id}_dashboard.json"
        with open(path, "w") as f:
            json.dump(dashboard_data, f, indent=2)

        return json.dumps(dashboard_data, indent=2)

    def _generate_pdf(self, report: ResearchReport) -> str:
        """Generate PDF (placeholder - would use weasyprint or similar)."""
        md = self._generate_markdown(report)
        # In production, convert markdown to PDF
        path = self.output_dir / f"{report.report_id}.pdf"
        with open(path, "w") as f:
            f.write(f"PDF placeholder for {report.title}\n\n{md}")

        return f"PDF saved to {path}"

    def _format_table(self, table: dict[str, Any]) -> str:
        headers = table.get("headers", [])
        rows = table.get("rows", [])

        if not headers:
            return ""

        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")

        return "\n".join(lines)

    def get_report(self, report_id: str) -> ResearchReport | None:
        with self._lock:
            return self._reports.get(report_id)

    def list_reports(self) -> list[ResearchReport]:
        with self._lock:
            return list(self._reports.values())
