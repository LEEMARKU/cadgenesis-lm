"""
cadgenesis.research.dashboard
=============================
Experiment dashboard for CADGenesis-LM research infrastructure.

Visualizes training, benchmarks, GPU/memory, datasets and model versions
as a self-contained HTML page with embedded JSON and lightweight JS charts
(no external assets).  ``ExperimentDashboard.render(tracker, registry,
benchmarks, system_stats) -> str``.
"""

from __future__ import annotations

import html
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _line_chart(title: str, series: Mapping[str, list[float]]) -> str:
    payload = json.dumps(series)
    return f"""
    <div class="card"><h2>{html.escape(title)}</h2>
    <canvas id="chart-{abs(hash(title)) % 10**8}"></canvas>
    <script>drawLine("{html.escape(title)}", {payload});</script></div>"""


TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>CADGenesis Research Dashboard</title>
<style>
body{{font-family:sans-serif;margin:0;background:#0f172a;color:#e2e8f0;}}
header{{padding:1rem 2rem;background:#1e293b;border-bottom:2px solid #334155;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
  gap:1rem;padding:1.5rem 2rem;}}
.card{{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:1rem;}}
h2{{font-size:.9rem;color:#94a3b8;margin:0 0 .5rem;text-transform:uppercase;}}
table{{width:100%;border-collapse:collapse;font-size:.8rem;}}
td,th{{border-bottom:1px solid #334155;padding:.3rem;text-align:left;}}
canvas{{width:100%;height:120px;}}
</style></head>
<body><header><h1>CADGenesis Research Dashboard</h1>
<div style="color:#94a3b8;font-size:.8rem">generated {timestamp}</div></header>
<div class="grid">{cards}</div>
<script>
function drawLine(title, series) {{
  const canvas = document.querySelectorAll("canvas")[
    document.querySelectorAll("canvas").length - 1];
  const ctx = canvas.getContext("2d");
  const keys = Object.keys(series);
  const all = keys.flatMap(k => series[k]);
  const max = Math.max(1, ...all), min = Math.min(0, ...all);
  const W = canvas.width = canvas.clientWidth, H = canvas.height = 120;
  ctx.fillStyle = "#0f172a"; ctx.fillRect(0, 0, W, H);
  const colors = ["#4ade80", "#60a5fa", "#facc15", "#f87171", "#c084fc"];
  keys.forEach((key, ki) => {{
    const values = series[key]; const n = values.length;
    ctx.strokeStyle = colors[ki % colors.length]; ctx.beginPath();
    values.forEach((v, i) => {{
      const x = (i / Math.max(1, n - 1)) * W;
      const y = H - ((v - min) / (max - min)) * (H - 8) - 4;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }});
    ctx.stroke();
  }});
}}
</script></body></html>"""


def _table(title: str, rows: Sequence[tuple[str, Any]]) -> str:
    body = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v)[:60])}</td></tr>"
        for k, v in rows
    )
    return f'<div class="card"><h2>{html.escape(title)}</h2><table>{body}</table></div>'


class ExperimentDashboard:
    """Renders the research dashboard from live experiment state."""

    def render(
        self,
        experiments: Iterable[Mapping[str, Any]] | None = None,
        datasets: Mapping[str, list[str]] | None = None,
        benchmarks: Mapping[str, float] | None = None,
        system_stats: Mapping[str, float] | None = None,
        model_versions: Mapping[str, list[str]] | None = None,
        training_curves: Mapping[str, list[float]] | None = None,
    ) -> str:
        cards: list[str] = []
        if training_curves:
            cards.append(_line_chart("Training loss", training_curves))
        if benchmarks:
            cards.append(_table("Benchmarks", list(benchmarks.items())))
        if system_stats:
            cards.append(_table("System (GPU/CPU/memory)", list(system_stats.items())))
        if model_versions:
            rows = [(name, ", ".join(versions[:5])) for name, versions in model_versions.items()]
            cards.append(_table("Model versions", rows))
        if datasets:
            rows = [(name, ", ".join(versions[:5])) for name, versions in datasets.items()]
            cards.append(_table("Datasets", rows))
        if experiments:
            rows = [
                (e.get("id", "?"), f"{e.get('status', '?')} | best={e.get('best_metric', '-')}")
                for e in experiments
            ]
            cards.append(_table("Experiments", rows))
        return TEMPLATE.format(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            cards="\n".join(cards) or '<div class="card">no data</div>',
        )


__all__ = ["ExperimentDashboard"]
