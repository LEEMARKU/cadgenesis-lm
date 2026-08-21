"""
cadgenesis.platform.dashboard
=============================
Operational dashboard for the CADGenesis-LM platform.

Generates a self-contained HTML dashboard (no JS framework) aggregating:

- inference: request rate, latency histogram, error count
- training: job status summary
- memory / GPU / CPU: gauges from the health checker + psutil (optional)
- agents: active agent fleet status (``AgentPlatform``)
- requests & failures: per-endpoint tallies from the metrics registry

``DashboardRenderer.render(metrics_registry, health_summary, ...) -> str``
produces the HTML; ``render_to_file`` writes it atomically.
"""

from __future__ import annotations

import html
import logging
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger("cadgenesis.platform.dashboard")

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CADGenesis-LM Operational Dashboard</title>
<style>
body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; 
      margin: 0; background: #0f172a; color: #e2e8f0; }}
header {{ padding: 1rem 2rem; background: #1e293b; border-bottom: 2px solid #334155; }}
h1 {{ margin: 0; font-size: 1.25rem; }} .sub {{ color: #94a3b8; font-size: .8rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 1rem; padding: 1.5rem 2rem; }}
.card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px;
  padding: 1rem; }}
.card h2 {{ font-size: .9rem; color: #94a3b8; margin: 0 0 .5rem;
  text-transform: uppercase; letter-spacing: .05em; }}
.metric {{ font-size: 1.6rem; font-weight: 600; }} .ok {{ color: #4ade80; }}
.warn {{ color: #facc15; }} .bad {{ color: #f87171; }}
table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
td, th {{ padding: .35rem .5rem; border-bottom: 1px solid #334155; text-align: left; }}
</style>
</head>
<body>
<header><h1>CADGenesis-LM Operational Dashboard</h1>
<div class="sub">generated {timestamp} &middot; {version}</div></header>
<div class="grid">
{cards}
</div>
</body>
</html>
"""


def _gauge_card(title: str, value: float, unit: str = "", status: str = "ok") -> str:
    css = {"ok": "ok", "warn": "warn", "bad": "bad"}.get(status, "ok")
    return (
        f'<div class="card"><h2>{html.escape(title)}</h2>'
        f'<div class="metric {css}">{value:.2f}{html.escape(unit)}</div></div>'
    )


def _table_card(title: str, rows: Sequence[tuple[str, str]]) -> str:
    body = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>" for k, v in rows
    )
    return f'<div class="card"><h2>{html.escape(title)}</h2><table>{body}</table></div>'


def _summary_card(title: str, rows: Sequence[tuple[str, str]]) -> str:
    return _table_card(title, rows)


class DashboardRenderer:
    """Renders the operational dashboard from live platform state."""

    def __init__(self, version: str = "6.0.0") -> None:
        self.version = version

    def render(
        self,
        metrics: Any = None,
        health: Mapping[str, Any] | None = None,
        system: Mapping[str, Any] | None = None,
        agents: Sequence[Mapping[str, Any]] | None = None,
        training_jobs: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        cards: list[str] = []

        # inference ----------------------------------------------------------
        if metrics is not None:
            snapshot = metrics.snapshot().get("metrics", []) if hasattr(metrics, "snapshot") else []
            by_name = {m.get("name", ""): m for m in snapshot}
            requests = by_name.get("cadgenesis.inference_requests", {}).get("value", 0)
            errors = by_name.get("cadgenesis.inference_errors", {}).get("value", 0)
            latency = by_name.get("cadgenesis.inference_latency", {})
            p95 = self._p95(latency)
            cards.append(_gauge_card("Inference requests", float(requests)))
            cards.append(_gauge_card("Failures", float(errors), status="bad" if errors else "ok"))
            cards.append(
                _gauge_card("Latency p95 (s)", float(p95), status="warn" if p95 > 5.0 else "ok")
            )
            cards.append(
                _summary_card(
                    "Endpoints",
                    [
                        ("GET /healthz", "liveness"),
                        ("GET /readyz", "readiness"),
                        ("GET /metrics", "Prometheus"),
                    ],
                )
            )

        # health --------------------------------------------------------------
        if health:
            status = health.get("status", "unknown")
            cards.append(
                _gauge_card("Platform status", 1.0 if status == "healthy" else 0.0, status=status)
            )
            cards.append(
                _summary_card(
                    "Health checks",
                    [
                        (c.get("name", "?"), str(c.get("ok", False)))
                        for c in health.get("checks", [])
                    ],
                )
            )

        # system (memory/CPU/GPU) ----------------------------------------------
        if system:
            cards.append(_gauge_card("CPU percent", float(system.get("cpu_percent", 0.0)), "%"))
            cards.append(
                _gauge_card(
                    "Memory percent",
                    float(system.get("memory_percent", 0.0)),
                    "%",
                    status="warn" if system.get("memory_percent", 0) > 85 else "ok",
                )
            )
            cards.append(_gauge_card("GPU utilization", float(system.get("gpu_util", 0.0)), "%"))
            cards.append(
                _gauge_card("GPU memory", float(system.get("gpu_memory_used_gb", 0.0)), " GiB")
            )

        # agents ----------------------------------------------------------------
        if agents is not None:
            cards.append(
                _summary_card(
                    "Agents",
                    [(a.get("name", "?"), str(a.get("state", "?"))) for a in agents[:12]],
                )
            )

        # training ----------------------------------------------------------------
        if training_jobs is not None:
            cards.append(
                _summary_card(
                    "Training jobs",
                    [(j.get("id", "?"), str(j.get("status", "?"))) for j in training_jobs[-8:]],
                )
            )

        return PAGE_TEMPLATE.format(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            version=html.escape(self.version),
            cards="\n".join(cards) if cards else '<div class="card"><h2>No data</h2></div>',
        )

    @staticmethod
    def _p95(histogram: Mapping[str, Any]) -> float:
        buckets = histogram.get("buckets", [])
        count = histogram.get("count", 0)
        if not buckets or count == 0:
            return 0.0
        target = 0.95 * count
        for bucket in buckets:
            if bucket.get("count", 0) >= target:
                bound = bucket.get("le", 0.0)
                return float(bound) if bound != float("inf") else 0.0
        return 0.0


def collect_system_stats() -> dict[str, Any]:
    """CPU/memory via psutil (optional); GPU via torch/nvidia-smi (optional)."""
    stats: dict[str, Any] = {
        "cpu_percent": 0.0,
        "memory_percent": 0.0,
        "gpu_util": 0.0,
        "gpu_memory_used_gb": 0.0,
    }
    try:  # pragma: no cover - optional
        import psutil  # type: ignore[import-not-found]

        stats["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        stats["memory_percent"] = psutil.virtual_memory().percent
    except ImportError:
        pass
    try:  # pragma: no cover - optional
        import torch

        if torch.cuda.is_available():
            stats["gpu_util"] = float(torch.cuda.utilization() or 0)
            stats["gpu_memory_used_gb"] = round(float(torch.cuda.memory_allocated() / (1024**3)), 2)
    except (ImportError, AttributeError):
        pass
    return stats


def render_to_file(
    path: str,
    metrics: Any = None,
    health: Mapping[str, Any] | None = None,
    system: Mapping[str, Any] | None = None,
    agents: Sequence[Mapping[str, Any]] | None = None,
    training_jobs: Sequence[Mapping[str, Any]] | None = None,
    version: str = "6.0.0",
) -> str:
    """Render the dashboard and write it atomically to ``path``."""
    renderer = DashboardRenderer(version=version)
    html_text = renderer.render(
        metrics=metrics, health=health, system=system, agents=agents, training_jobs=training_jobs
    )
    target_dir = os.path.dirname(path) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target_dir, prefix=".dashboard-", suffix=".html")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(html_text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    logger.info("dashboard rendered: %s", path)
    return path


__all__ = ["DashboardRenderer", "collect_system_stats", "render_to_file"]
