"""
cadgenesis.research.reports
===========================
Automated report generator for CADGenesis-LM research infrastructure.

- Markdown: primary format (always available)
- HTML: styled single-file reports (always available)
- PDF: via optional ``reportlab`` (fails gracefully with a clear message)
- Interactive dashboard: self-contained HTML with embedded JSON data

``ReportBuilder`` collects experiment/benchmark/ablation artifacts and
renders them in any supported format.
"""

from __future__ import annotations

import html
import json
import logging
import time
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger("cadgenesis.research.reports")

FORMATS = ("markdown", "html", "pdf", "dashboard")


class ReportBuilder:
    """Assemble sections from experiments, benchmarks and comparisons."""

    def __init__(self, title: str = "CADGenesis-LM Research Report") -> None:
        self.title = title
        self.sections: list[tuple[str, Any]] = []  # (title, data)

    def add_section(self, title: str, data: Any) -> ReportBuilder:
        self.sections.append((title, data))
        return self

    # ---------------------------------------------------------- rendering

    def render_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}_")
        lines.append("")
        for title, data in self.sections:
            lines.append(f"## {title}")
            lines.append("")
            if isinstance(data, Mapping):
                for key, value in data.items():
                    if isinstance(value, (list, dict)):
                        lines.append(f"- **{key}**: `{json.dumps(value, default=str)[:200]}`")
                    else:
                        lines.append(f"- **{key}**: {value}")
            else:
                lines.append(str(data))
            lines.append("")
        return "\n".join(lines)

    def render_html(self) -> str:
        rows: list[str] = []
        for title, data in self.sections:
            if isinstance(data, Mapping):
                body = "".join(
                    "<tr><td>"
                    f"{html.escape(str(k))}</td><td>"
                    f"{html.escape(json.dumps(v, default=str)[:300])}</td></tr>"
                    for k, v in data.items()
                )
                rows.append(
                    f"<section><h2>{html.escape(title)}</h2>"
                    f'<table class="data">{body}</table></section>'
                )
            else:
                rows.append(
                    f"<section><h2>{html.escape(title)}</h2>"
                    f"<pre>{html.escape(str(data))}</pre></section>"
                )
        return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{html.escape(self.title)}</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;}}
h1{{border-bottom:2px solid #333;padding-bottom:.25rem;}}
table{{border-collapse:collapse;width:100%;margin:.5rem 0;}}
td,th{{border:1px solid #ccc;padding:.3rem .5rem;text-align:left;font-size:.85rem;}}
section{{margin:1rem 0;}}</style></head>
<body><h1>{html.escape(self.title)}</h1>{"".join(rows)}</body></html>"""

    def render_pdf(self, path: str) -> str:
        try:
            from reportlab.lib.pagesizes import A4  # type: ignore[import-not-found]
            from reportlab.lib.styles import getSampleStyleSheet  # type: ignore[import-not-found]
            from reportlab.platypus import (  # type: ignore[import-not-found]
                Paragraph,
                SimpleDocTemplate,
                Spacer,
            )
        except ImportError as exc:
            raise RuntimeError("PDF rendering requires 'reportlab'") from exc
        doc = SimpleDocTemplate(path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = [Paragraph(self.title, styles["Title"]), Spacer(1, 12)]
        for title, data in self.sections:
            story.append(Paragraph(title, styles["Heading2"]))
            if isinstance(data, Mapping):
                for key, value in data.items():
                    rendered = html.escape(json.dumps(value, default=str)[:300])
                    story.append(
                        Paragraph(
                            f"<b>{html.escape(str(key))}:</b> {rendered}",
                            styles["BodyText"],
                        )
                    )
            else:
                story.append(Paragraph(html.escape(str(data))[:1000], styles["BodyText"]))
            story.append(Spacer(1, 8))
        doc.build(story)
        return path

    def render_dashboard(self) -> str:
        """Interactive dashboard: embedded JSON + minimal JS filtering."""
        payload = json.dumps(
            {
                "title": self.title,
                "generated": time.time(),
                "sections": [{"title": t, "data": d} for t, d in self.sections],
            },
            default=str,
        )
        return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{html.escape(self.title)}</title>
<style>body{{font-family:sans-serif;margin:2rem;}}
.card{{border:1px solid #ccc;border-radius:6px;padding:1rem;margin:.5rem 0;}}
h2{{margin:0 0 .5rem;}} pre{{white-space:pre-wrap;font-size:.8rem;}}</style></head>
<body><h1>{html.escape(self.title)}</h1>
<input placeholder="filter..." oninput="filterData(this.value)">
<div id="root"></div>
<script>
const DATA = {payload};
function filterData(q) {{
  q = (q || "").toLowerCase();
  document.getElementById("root").innerHTML = DATA.sections
    .filter(s => JSON.stringify(s).toLowerCase().includes(q))
    .map(s => `<div class="card"><h2>${{s.title}}</h2>
<pre>${{JSON.stringify(s.data, null, 2)}}</pre></div>`)
    .join("");
}}
filterData("");
</script></body></html>"""

    def render(self, format: str = "markdown", path: str | None = None) -> str:
        if format == "markdown":
            text = self.render_markdown()
        elif format == "html":
            text = self.render_html()
        elif format == "pdf":
            if path is None:
                raise ValueError("PDF rendering requires an output path")
            return self.render_pdf(path)
        elif format == "dashboard":
            text = self.render_dashboard()
        else:
            raise ValueError(f"unknown report format {format!r}; expected {FORMATS}")
        if path:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
        return text


def experiment_report(
    tracker: Any, experiment_id: str, builder: ReportBuilder | None = None
) -> ReportBuilder:
    """Convenience: build a report from one tracked experiment."""
    record = tracker.get(experiment_id)
    if record is None:
        raise KeyError(f"unknown experiment {experiment_id!r}")
    builder = builder or ReportBuilder(title=f"Experiment {experiment_id}")
    builder.add_section("Metadata", record.to_dict())
    return builder


__all__ = ["FORMATS", "ReportBuilder", "experiment_report"]
