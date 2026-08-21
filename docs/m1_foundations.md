# M1 — Foundations: Utilities & Observability

Milestone M1 delivers the horizontal foundation every other subsystem builds
on: shared helpers, structured logging, metrics/tracing telemetry, and health /
drift / alert monitoring.  It advances pillars **11 (Production Platform)** and
**12 (Research Infrastructure)**.

## Scope

| Package | Module | Deliverables |
|---------|--------|--------------|
| `cadgenesis.utils` | `decorators.py` | `timed`, `retry`, `memoize`, `singleton`, `deprecated`, `synchronized`, `log_calls`, `classmethod_dispatch` |
| | `filesystem.py` | `ensure_dir`, `atomic_write_text/bytes`, `safe_join`, `iter_files`, `human_readable_size`, `temp_dir`, `file_lock`, `copy_tree` |
| | `hashing.py` | `sha256_file`, `md5_file`, `content_hash`, `stable_hash`, `fingerprint`, `deduplicate_paths`, `verify_artifact` |
| | `math.py` | `clamp`, `lerp`, `safe_div`, `smoothstep`, `RunningStats`, `ExponentialMovingAverage`, `percentile`, `median`, geometry helpers |
| | `time.py` | `utc_now`, `now_iso`, `format_duration`, `parse_duration`, `Stopwatch`, `RateLimiter` |
| `cadgenesis.logging` | `config.py` | `LoggingConfig`, `setup_logging`, `get_logger` (console / rotating-file / JSON sinks) |
| | `emitter.py` | `StructuredLogEmitter`, `emit` (structured key/value records) |
| `cadgenesis.telemetry` | `metrics.py` | `Counter`, `Gauge`, `Histogram`, `MetricsRegistry`, `StepTimer` |
| | `tracing.py` | `Tracer`, `Span`, `SpanContext`, span context managers + decorator |
| | `logs.py` | `TelemetryLogger`, `EventBuffer`, `TelemetryEvent`, `log_event` |
| `cadgenesis.monitoring` | `health.py` | `HealthChecker`, `HealthResult`, `HealthStatus`, memory/disk checks |
| | `drift.py` | PSI/KL/JS `compute_drift`, `FeatureDriftMonitor`, `DriftReport` |
| | `alerts.py` | `AlertManager`, `AlertRule`, `ThresholdRule`, `AlertSeverity`, handlers |
| `cadgenesis.config` | `cad_config.py` | New `ObservabilityConfig` sub-config wired into `CADConfig` |

## Design principles

- **Single source of truth:** all knobs live in `ObservabilityConfig` /
  `LoggingConfig`; nothing is hard-coded in the helpers.
- **Thread safety:** metrics, drift monitors, alert managers, rate limiters and
  the singleton decorator are lock-protected.
- **Determinism:** `content_hash` / `stable_hash` sort dict keys, so digests are
  reproducible across runs and platforms (used for checkpoint fingerprinting).
- **No-op safe:** tracing/metrics can be globally disabled without changing
  call sites (`set_tracing_enabled`, `TelemetryLogger(capture=False)`).
- **Atomicity:** filesystem writes go through temp-file + rename; `safe_join`
  rejects path traversal.

## Configuration

`ObservabilityConfig` is part of `CADConfig`:

```python
from cadgenesis.config import CADConfig

cfg = CADConfig()
cfg.observability.log_level = "INFO"
cfg.observability.tracing_enabled = True
```

`setup_logging` reads environment variables (`LOG_LEVEL`) by default and is
idempotent.

## Usage examples

```python
from cadgenesis.utils import retry, content_hash, Stopwatch
from cadgenesis.telemetry import MetricsRegistry, Tracer
from cadgenesis.monitoring import HealthChecker, FeatureDriftMonitor, AlertManager, ThresholdRule


@retry(attempts=3)
def build(): ...


with Stopwatch() as sw:
    build()
sw.elapsed_str

registry = MetricsRegistry()
registry.counter("inference.requests").inc()
registry.snapshot()

tracer = Tracer()
with tracer.span("forward"):
    ...

checker = HealthChecker()
checker.register("memory", check_memory_usage)
checker.summary()

alerts = AlertManager()
alerts.add_rule(ThresholdRule("high_loss", "loss", 5.0))
alerts.evaluate({"loss": 6.0})
```

## Testing

- `tests/utils/` — decorators, filesystem, hashing, math, time (75 tests)
- `tests/logging/` — config + structured emitter (12 tests)
- `tests/telemetry/` — metrics, tracing, logs (28 tests)
- `tests/monitoring/` — health, drift, alerts (24 tests)

Run: `python -m pytest tests/utils tests/logging tests/telemetry tests/monitoring -q`

## Verification

- `python scripts/audit_repo.py` — stub count reduced from 129 to 116; the
  foundation pillar modules now report no stubs.
- `python -m ruff check src/cadgenesis/utils src/cadgenesis/logging \
  src/cadgenesis/telemetry src/cadgenesis/monitoring src/cadgenesis/config \
  tests/utils tests/logging tests/telemetry tests/monitoring` — clean.
- Full suite: `python -m pytest -q` → 428 passed (was 300 at baseline).
