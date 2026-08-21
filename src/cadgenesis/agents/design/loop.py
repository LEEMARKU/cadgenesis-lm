"""cadgenesis.agents.design.loop
===============================
Autonomous design-orchestration loop (Pillar 5).

:class:`DesignOrchestrationLoop` runs a closed-loop, self-correcting design
workflow:

1. :class:`~cadgenesis.agents.design.fea.FEAStressAgent` evaluates the von
   Mises stress of the part under every load case.
2. When the factor of safety is below the yield target
   (``sigma_vm * target > sigma_yield``), the FEA agent reinforces the
   cross-section and the loop re-analyzes (bounded by ``max_iterations``).
3. When structurally sound, :class:`~cadgenesis.agents.design.dfm.DFMManufacturingAgent`
   checks manufacturability and — if the process is not viable — recommends
   and switches to an alternative process.
4. :class:`~cadgenesis.agents.design.cost_estimator.CostEstimatorAgent`
   prices the design and the loop records the cost of every iteration.

The loop converges when a step passes structurally *and* is manufacturable.
:class:`DesignSwarm` bundles the four role agents with the loop, and
:func:`build_design_swarm` assembles them with one call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from cadgenesis.agents.base import AgentRequest
from cadgenesis.agents.design.cost_estimator import CostEstimatorAgent
from cadgenesis.agents.design.dfm import DFMManufacturingAgent
from cadgenesis.agents.design.fea import FEAStressAgent
from cadgenesis.agents.design.lead import LeadArchitectAgent

_LOGGER = logging.getLogger("cadgenesis.agents.design")


@dataclass
class DesignIteration:
    """The outcome of one loop step."""

    index: int
    fea: dict[str, Any]
    reinforcement: dict[str, Any] | None = None
    dfm: dict[str, Any] | None = None
    cost: dict[str, Any] | None = None
    process: str = "machining"
    passed: bool = False
    retryable: bool = False
    message: str = ""
    state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "fea": dict(self.fea),
            "reinforcement": dict(self.reinforcement) if self.reinforcement else None,
            "dfm": dict(self.dfm) if self.dfm else None,
            "cost": dict(self.cost) if self.cost else None,
            "process": self.process,
            "passed": self.passed,
            "retryable": self.retryable,
            "message": self.message,
        }


@dataclass
class DesignReport:
    """The result of a full autonomous design run."""

    goal: str
    converged: bool
    iterations: list[DesignIteration]
    final_parameters: dict[str, Any]
    final_material: Any
    final_process: str
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "converged": self.converged,
            "iterations": [iteration.to_dict() for iteration in self.iterations],
            "final_parameters": dict(self.final_parameters),
            "final_material": self.final_material,
            "final_process": self.final_process,
            "summary": dict(self.summary),
        }


class DesignOrchestrationLoop:
    """Closed-loop stress / DFM / cost design workflow."""

    def __init__(
        self,
        fea: FEAStressAgent | None = None,
        dfm: DFMManufacturingAgent | None = None,
        cost: CostEstimatorAgent | None = None,
        max_iterations: int = 10,
        target_safety_factor: float = 1.5,
        process_switching: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if target_safety_factor <= 1.0:
            raise ValueError("target_safety_factor must be > 1.0")
        self.fea = fea or FEAStressAgent()
        self.dfm = dfm or DFMManufacturingAgent()
        self.cost = cost or CostEstimatorAgent()
        self.max_iterations = max_iterations
        self.target_safety_factor = target_safety_factor
        self.process_switching = process_switching
        self._logger = logger or _LOGGER
        self.state: dict[str, Any] = {}
        self.iterations: list[DesignIteration] = []

    # ------------------------------------------------------------- workflow

    def run(self, task: dict[str, Any]) -> DesignReport:
        """Execute the autonomous loop for a design task.

        ``task`` must contain a ``part`` dict (``feature``, ``parameters``,
        optional ``material``/``name``) and may carry ``load_cases``,
        ``target_safety_factor``, ``process``, ``goal`` and ``quantity``.
        """
        part = task.get("part")
        if not isinstance(part, dict):
            raise ValueError("task requires a 'part' dictionary")
        self.state = {
            "part": dict(part),
            "load_cases": list(task.get("load_cases", [])),
            "target_safety_factor": float(
                task.get("target_safety_factor", self.target_safety_factor)
            ),
            "process": str(task.get("process", "machining")),
            "goal": str(task.get("goal", "design a part")),
            "quantity": int(task.get("quantity", 1)),
            "iteration": 0,
        }
        self.iterations = []
        for _ in range(self.max_iterations):
            iteration = self.step(self.state)
            self.iterations.append(iteration)
            self.state = iteration.state
            if iteration.passed:
                self._logger.info("design converged on iteration %d", iteration.index + 1)
                break
            if not iteration.retryable:
                self._logger.info(
                    "design blocked after iteration %d: %s", iteration.index + 1, iteration.message
                )
                break
        return self._report()

    def step(self, state: dict[str, Any]) -> DesignIteration:
        """Advance the design by one stress/DFM/cost cycle."""
        part = dict(state.get("part", {}))
        load_cases = list(state.get("load_cases", []))
        target = float(state.get("target_safety_factor", self.target_safety_factor))
        process = str(state.get("process", "machining"))
        index = int(state.get("iteration", 0))
        quantity = int(state.get("quantity", 1))

        fea_result = self.fea.handle(
            AgentRequest(
                "fea_stress",
                "analyze",
                {"object": part, "load_cases": load_cases, "target_safety_factor": target},
            )
        )
        fea_output = fea_result.output if fea_result.ok else {}
        safety_factor = float(fea_output.get("factor_of_safety", 0.0))
        fea_passed = fea_result.ok and safety_factor >= target

        if not fea_passed:
            reinforce_result = self.fea.handle(
                AgentRequest(
                    "fea_stress",
                    "reinforce",
                    {
                        "object": part,
                        "target_safety_factor": target,
                        "current_safety_factor": safety_factor,
                    },
                )
            )
            reinforcement = reinforce_result.output if reinforce_result.ok else {}
            growth = float(reinforcement.get("growth_factor", 1.0))
            if growth > 1.0:
                next_part = {**part, "parameters": dict(reinforcement["parameters"])}
                next_state = {**state, "part": next_part, "iteration": index + 1}
                return DesignIteration(
                    index=index,
                    fea=fea_output,
                    reinforcement=reinforcement,
                    process=process,
                    passed=False,
                    retryable=True,
                    message=(
                        f"factor of safety {safety_factor:.2f} below target "
                        f"{target:.2f}; cross-section reinforced x{growth:.3f}"
                    ),
                    state=next_state,
                )
            return DesignIteration(
                index=index,
                fea=fea_output,
                reinforcement=reinforcement,
                process=process,
                passed=False,
                retryable=False,
                message=(
                    f"factor of safety {safety_factor:.2f} below target {target:.2f}; "
                    "feature is not reinforceable"
                ),
                state=state,
            )

        # Structural pass -> DFM gate (with optional process switching).
        dfm_part = {"name": part.get("name", "part"), **dict(part.get("parameters", {}))}
        dfm_result = self.dfm.handle(
            AgentRequest("dfm_manufacturing", "assess", {"part": dfm_part, "processes": [process]})
        )
        dfm_output = dfm_result.output if dfm_result.ok else {}
        dfm_passed = dfm_result.ok
        switched_to: str | None = None
        if not dfm_passed and self.process_switching:
            recommendation = self.dfm.handle(
                AgentRequest("dfm_manufacturing", "recommend_process", {"part": dfm_part})
            )
            if recommendation.ok:
                recommended = recommendation.output.get("recommended")
                if recommended and recommended != process:
                    switched_to = str(recommended)
                    process = switched_to
                    dfm_result = self.dfm.handle(
                        AgentRequest(
                            "dfm_manufacturing",
                            "assess",
                            {"part": dfm_part, "processes": [process]},
                        )
                    )
                    dfm_output = dfm_result.output if dfm_result.ok else dfm_output
                    dfm_passed = dfm_result.ok

        cost_result = self.cost.handle(
            AgentRequest(
                "cost_estimator",
                "estimate",
                {"part": part, "process": process, "quantity": quantity},
            )
        )
        cost_output = cost_result.output if cost_result.ok else {}
        total_cost = float(cost_output.get("total_cost_usd", 0.0))

        passed = dfm_passed
        retryable = not passed and switched_to is not None
        next_state = {**state, "part": part, "process": process, "iteration": index + 1}
        return DesignIteration(
            index=index,
            fea=fea_output,
            dfm=dfm_output,
            cost=cost_output,
            process=process,
            passed=passed,
            retryable=retryable,
            message=(
                f"safety factor {safety_factor:.2f} OK; DFM "
                f"{'passed' if dfm_passed else 'failed'} for {process}; "
                f"cost ${total_cost:.2f}"
            ),
            state=next_state,
        )

    # ---------------------------------------------------------------- report

    def _report(self) -> DesignReport:
        final_part = dict(self.state.get("part", {}))
        final_iteration = self.iterations[-1] if self.iterations else None
        last_fea = (final_iteration.fea if final_iteration else None) or {}
        last_cost = (final_iteration.cost if final_iteration else None) or {}
        summary = {
            "iterations": len(self.iterations),
            "max_iterations": self.max_iterations,
            "final_safety_factor": float(last_fea.get("factor_of_safety", 0.0)),
            "final_sigma_vm_mpa": float(last_fea.get("sigma_vm_mpa", 0.0)),
            "final_sigma_yield_mpa": float(last_fea.get("sigma_yield_mpa", 0.0)),
            "total_cost_usd": float(last_cost.get("total_cost_usd", 0.0)),
            "reinforcement_steps": sum(
                1 for iteration in self.iterations if iteration.reinforcement
            ),
        }
        return DesignReport(
            goal=str(self.state.get("goal", "")),
            converged=bool(final_iteration and final_iteration.passed),
            iterations=list(self.iterations),
            final_parameters=dict(final_part.get("parameters", {})),
            final_material=final_part.get("material"),
            final_process=str(self.state.get("process", "machining")),
            summary=summary,
        )


@dataclass
class DesignSwarm:
    """The design-swarm team: four role agents plus their shared loop."""

    lead: LeadArchitectAgent
    fea: FEAStressAgent
    dfm: DFMManufacturingAgent
    cost: CostEstimatorAgent
    loop: DesignOrchestrationLoop

    def register_all(self, registry: Any) -> None:
        """Register every swarm agent into an :class:`AgentRegistry`."""
        for agent in (self.lead, self.fea, self.dfm, self.cost):
            registry.register(agent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agents": [agent.describe() for agent in (self.lead, self.fea, self.dfm, self.cost)],
            "max_iterations": self.loop.max_iterations,
            "target_safety_factor": self.loop.target_safety_factor,
        }


def build_design_swarm(
    fea: FEAStressAgent | None = None,
    dfm: DFMManufacturingAgent | None = None,
    cost: CostEstimatorAgent | None = None,
    max_iterations: int = 10,
    target_safety_factor: float = 1.5,
    process_switching: bool = True,
) -> DesignSwarm:
    """Assemble a fully wired :class:`DesignSwarm`."""
    loop = DesignOrchestrationLoop(
        fea=fea,
        dfm=dfm,
        cost=cost,
        max_iterations=max_iterations,
        target_safety_factor=target_safety_factor,
        process_switching=process_switching,
    )
    return DesignSwarm(
        lead=LeadArchitectAgent(loop=loop),
        fea=loop.fea,
        dfm=loop.dfm,
        cost=loop.cost,
        loop=loop,
    )


__all__ = [
    "DesignIteration",
    "DesignOrchestrationLoop",
    "DesignReport",
    "DesignSwarm",
    "build_design_swarm",
]
