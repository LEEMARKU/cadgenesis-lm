"""
cadgenesis.research.ablation
============================
Ablation framework for CADGenesis-LM research infrastructure.

Runs a baseline and a series of component-disabled variants, then reports
the delta per ablated component.  Supported ablation targets:

- component: disable a named module/component of a pipeline
- layer: prune/disable transformer layers by index
- attention: disable attention head types (self/geometry/constraint/memory/
  agent/uncertainty)
- memory: disable memory pools or retrieval
- agent: disable agent roles in a fleet

``AblationSpec`` declaratively describes one ablation; ``run_ablation``
executes baseline + variants through an injected ``runner`` callable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("cadgenesis.research.ablation")

Runner = Callable[[dict[str, Any]], dict[str, Any]]

ABLATION_KINDS = ("component", "layer", "attention", "memory", "agent")


@dataclass
class AblationSpec:
    """One ablation variant: which component is disabled and how."""

    kind: str  # component | layer | attention | memory | agent
    target: str  # e.g. "attention.geometry" | "0" (layer index) | "lora" ...
    label: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target": self.target,
            "label": self.label,
            "config": dict(self.config),
        }


@dataclass
class AblationResult:
    """Baseline-vs-variant comparison for one ablation."""

    spec: AblationSpec
    baseline: dict[str, Any]
    variant: dict[str, Any]
    deltas: dict[str, float] = field(default_factory=dict)

    def compute_deltas(self, metric_keys: Sequence[str] | None = None) -> dict[str, float]:
        keys = metric_keys or sorted(set(self.baseline) & set(self.variant))
        self.deltas = {
            key: float(self.variant.get(key, 0.0)) - float(self.baseline.get(key, 0.0))
            for key in keys
            if isinstance(self.baseline.get(key), (int, float))
        }
        return self.deltas

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "baseline": dict(self.baseline),
            "variant": dict(self.variant),
            "deltas": dict(self.deltas),
        }


class AblationEngine:
    """Runs baseline + variants and reports per-component impact."""

    def __init__(
        self,
        runner: Runner,
        metric_keys: Sequence[str] | None = None,
        baseline_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.runner = runner
        self.metric_keys = metric_keys
        self.baseline_config = dict(baseline_config or {})

    def run(self, specs: Sequence[AblationSpec]) -> list[AblationResult]:
        logger.info("ablation: running baseline")
        baseline = self.runner(dict(self.baseline_config))
        results: list[AblationResult] = []
        for spec in specs:
            logger.info("ablation: %s/%s", spec.kind, spec.target)
            variant_config = self._apply(dict(self.baseline_config), spec)
            variant = self.runner(variant_config)
            result = AblationResult(spec=spec, baseline=baseline, variant=variant)
            result.compute_deltas(self.metric_keys)
            results.append(result)
        return results

    @staticmethod
    def _apply(config: dict[str, Any], spec: AblationSpec) -> dict[str, Any]:
        """Encode an ablation into a config dict (kind-specific mutation)."""
        mutated = dict(config)
        if spec.kind == "component":
            mutated[f"ablate.{spec.target}"] = True
        elif spec.kind == "layer":
            layers = list(mutated.get("ablate.layers", []))
            layers.append(spec.target)
            mutated["ablate.layers"] = layers
        elif spec.kind == "attention":
            heads = dict(mutated.get("model.heads", {}))
            heads[spec.target] = 0
            mutated["model.heads"] = heads
        elif spec.kind == "memory":
            mutated["memory.disabled"] = spec.target
        elif spec.kind == "agent":
            agents = list(mutated.get("ablate.agents", []))
            agents.append(spec.target)
            mutated["ablate.agents"] = agents
        else:
            raise ValueError(f"unknown ablation kind {spec.kind!r}; expected {ABLATION_KINDS}")
        mutated.update(spec.config)
        return mutated

    def summary(self, results: Sequence[AblationResult]) -> dict[str, Any]:
        return {
            "count": len(results),
            "results": [r.to_dict() for r in results],
            "most_impactful": self._most_impactful(results),
        }

    @staticmethod
    def _most_impactful(results: Sequence[AblationResult]) -> dict[str, Any] | None:
        scored: list[tuple[float, AblationResult]] = []
        for result in results:
            magnitude = sum(abs(v) for v in result.deltas.values())
            scored.append((magnitude, result))
        if not scored:
            return None
        magnitude, best = max(scored, key=lambda pair: pair[0])
        return {
            "target": best.spec.target,
            "kind": best.spec.kind,
            "magnitude": round(magnitude, 6),
        }


__all__ = ["ABLATION_KINDS", "AblationEngine", "AblationResult", "AblationSpec"]
