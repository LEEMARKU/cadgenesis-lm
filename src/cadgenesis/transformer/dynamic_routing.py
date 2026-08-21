"""
cadgenesis.transformer.dynamic_routing
======================================
Dynamic Computation Routing for CADGenesis-LM v6.0 (Pillar 1).

Adaptive computation lets the model spend FLOPs *only where they are needed*.
Two complementary mechanisms are provided:

1. :class:`ComputationBudget` — a hard cap on the number of transformer layers
   executed, expressed as a fraction of the total depth.  ``budget=1.0`` runs
   the full model; ``budget=0.5`` halves depth at a predictable latency win.

2. :class:`EarlyExitGate` — a *confidence-triggered* early stop: once the
   model's uncertainty head reports a confidence above ``threshold`` the
   remaining layers are skipped for that sequence.

3. :class:`DynamicRoutingController` composes the two into the decision loop
   used by :class:`cadgenesis.transformer.hierarchical_transformer.HierarchicalCADTransformer`
   and by the config-driven builder in
   :mod:`cadgenesis.transformer.evolution`.  It is deliberately a *pure
   controller* (no parameters) so it can be tested and reused anywhere.

The existing :class:`cadgenesis.transformer.self_designing.routing.DynamicLayerRouter`
already provides per-token *adaptive routing / dynamic depth* (Gumbel-sigmoid
skip masks); this module adds the missing **early exit** and **computation
budgeting** capabilities.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


class ComputationBudget:
    """
    Hard per-sequence layer budget.

    Parameters
    ----------
    budget : float
        Fraction of total layers to execute, in ``[0, 1]``.
    """

    def __init__(self, budget: float = 1.0):
        if not 0.0 <= budget <= 1.0:
            raise ValueError(f"budget must be in [0, 1]; got {budget}.")
        self.budget = budget

    @property
    def active_fraction(self) -> float:
        return self.budget

    def max_layers(self, total_layers: int) -> int:
        """Maximum number of layers to run out of ``total_layers``."""
        if total_layers < 1:
            raise ValueError("total_layers must be >= 1")
        return max(1, math.ceil(self.budget * total_layers))

    def report(self) -> dict:
        return {
            "budget": self.budget,
            "active_fraction": self.budget,
        }


class EarlyExitGate:
    """
    Confidence-thresholded early exit decision maker.

    Parameters
    ----------
    threshold : float
        Confidence in ``(0, 1]`` that triggers an early exit.  ``0`` disables
        early exit entirely (backward-compatible default).
    """

    def __init__(self, threshold: float = 0.0):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1]; got {threshold}.")
        self.threshold = threshold

    @property
    def enabled(self) -> bool:
        return self.threshold > 0.0

    def should_exit(
        self,
        step_idx: int,
        confidence: float,
        *,
        budget_cap: int,
        min_steps: int = 1,
    ) -> bool:
        """
        Decide whether to stop after ``step_idx`` (0-based).

        ``budget_cap`` is the layer count enforced by :class:`ComputationBudget`;
        ``min_steps`` guarantees at least that many layers always run.
        """
        if budget_cap < 1:
            raise ValueError("budget_cap must be >= 1")
        if min_steps < 1:
            raise ValueError("min_steps must be >= 1")
        if step_idx < min_steps - 1:
            return False
        if step_idx >= budget_cap - 1:
            return True
        return self.enabled and confidence >= self.threshold

    def report(self) -> dict:
        return {"threshold": self.threshold, "enabled": self.enabled}


class DynamicRoutingController:
    """
    Composes :class:`ComputationBudget` and :class:`EarlyExitGate` into the
    layer-loop decision used by hierarchical / budgeted forward passes.

    Parameters
    ----------
    total_layers : int
        Total number of sequential layers available.
    budget : float
        Computation budget fraction in ``[0, 1]``.
    early_exit_threshold : float
        Confidence threshold that triggers early exit (0 disables).
    min_steps : int
        Minimum number of layers always executed.
    """

    def __init__(
        self,
        total_layers: int,
        budget: float = 1.0,
        early_exit_threshold: float = 0.0,
        min_steps: int = 1,
    ):
        if total_layers < 1:
            raise ValueError("total_layers must be >= 1")
        if min_steps < 1 or min_steps > total_layers:
            raise ValueError(f"min_steps must be in [1, {total_layers}]")
        self.total_layers = total_layers
        self.min_steps = min_steps
        self.budget = ComputationBudget(budget)
        self.early_exit = EarlyExitGate(early_exit_threshold)
        self.reset()

    @property
    def max_layers(self) -> int:
        """Number of layers the budget allows for this sequence."""
        return self.budget.max_layers(self.total_layers)

    def reset(self) -> None:
        """Reset per-forward telemetry."""
        self.exit_layer: int | None = None
        self.exit_reason: str = "completed"
        self.layers_executed: int = 0

    def should_stop(
        self,
        step_idx: int,
        confidence: float | None = None,
        *,
        done: bool = False,
    ) -> bool:
        """
        Given the 0-based layer index ``step_idx``, decide whether to stop.

        ``confidence`` may be None when the model has no uncertainty head (the
        budget cap still applies).  Returns True when the loop must terminate
        *before processing layer ``step_idx + 1``*.
        """
        cap = self.max_layers
        if done or step_idx >= cap - 1:
            self.exit_layer = step_idx
            self.exit_reason = "budget" if not done else "done"
            self.layers_executed = step_idx + 1
            return True
        if confidence is not None and self.early_exit.should_exit(
            step_idx, confidence, budget_cap=cap, min_steps=self.min_steps
        ):
            self.exit_layer = step_idx
            self.exit_reason = "early_exit"
            self.layers_executed = step_idx + 1
            return True
        return False

    def report(self) -> dict:
        """Telemetry for profiling / experiment bookkeeping."""
        return {
            "total_layers": self.total_layers,
            "max_layers": self.max_layers,
            "exit_layer": self.exit_layer,
            "exit_reason": self.exit_reason,
            "layers_executed": self.layers_executed,
            "savings_fraction": round(1.0 - (self.layers_executed / max(self.total_layers, 1)), 4),
            "budget": self.budget.report(),
            "early_exit": self.early_exit.report(),
        }
