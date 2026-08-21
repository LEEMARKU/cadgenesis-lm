"""cadgenesis.agents.design
=========================
Autonomous design swarm (Pillar 5).

The design swarm is a self-correcting multi-agent team:

* :class:`LeadArchitectAgent` — runs the autonomous design loop.
* :class:`FEAStressAgent` — von Mises stress evaluation and automatic
  cross-section reinforcement when the safety factor drops below the yield
  target.
* :class:`DFMManufacturingAgent` — DFM checks and process ranking.
* :class:`CostEstimatorAgent` — per-iteration manufacturing cost tracking.
* :class:`DesignOrchestrationLoop` / :class:`DesignSwarm` — the loop and the
  assembled team (see :func:`build_design_swarm`).
"""

from cadgenesis.agents.design.cost_estimator import CostEstimatorAgent
from cadgenesis.agents.design.dfm import DFMManufacturingAgent
from cadgenesis.agents.design.fea import FEAStressAgent, ReinforcementPolicy
from cadgenesis.agents.design.lead import LeadArchitectAgent
from cadgenesis.agents.design.loop import (
    DesignIteration,
    DesignOrchestrationLoop,
    DesignReport,
    DesignSwarm,
    build_design_swarm,
)

__all__ = [
    "CostEstimatorAgent",
    "DFMManufacturingAgent",
    "DesignIteration",
    "DesignOrchestrationLoop",
    "DesignReport",
    "DesignSwarm",
    "FEAStressAgent",
    "LeadArchitectAgent",
    "ReinforcementPolicy",
    "build_design_swarm",
]
