"""tests/agents/test_design_swarm.py
===================================
Unit tests for the autonomous design swarm (Pillar 5): the four role agents
(LeadArchitect / FEAStress / DFMManufacturing / CostEstimator), the closed
stress-reinforcement loop and its CADConfig wiring.
"""

from __future__ import annotations

import pytest

from cadgenesis.agents.base import AgentRequest
from cadgenesis.agents.design import (
    CostEstimatorAgent,
    DesignOrchestrationLoop,
    DFMManufacturingAgent,
    FEAStressAgent,
    LeadArchitectAgent,
    ReinforcementPolicy,
    build_design_swarm,
)
from cadgenesis.agents.registry import AgentRegistry
from cadgenesis.config.cad_config import CADConfig

_BRACKET = {
    "name": "bracket",
    "feature": "block",
    "parameters": {"length": 40, "width": 30, "height": 50},
    "material": "steel",
}
_HEAVY_LOAD = {
    "name": "heavy",
    "conditions": [{"kind": "force", "magnitude": 300000.0}],
}
_LIGHT_LOAD = {
    "name": "light",
    "conditions": [{"kind": "force", "magnitude": 5000.0}],
}


def _analyze(fea: FEAStressAgent, part: dict, load: dict) -> dict:
    result = fea.process(
        AgentRequest(
            "fea_stress",
            "analyze",
            {"object": part, "load_cases": [load], "target_safety_factor": 2.0},
        )
    )
    assert "sigma_vm_mpa" in result.output
    return result


# ------------------------------------------------------------------ FEA agent


def test_fea_analyze_detects_overstress():
    fea = FEAStressAgent()
    result = _analyze(fea, _BRACKET, _HEAVY_LOAD)
    assert not result.ok
    assert not result.output["passed"]
    assert result.output["factor_of_safety"] < 2.0
    assert (
        result.output["sigma_vm_mpa"] * result.output["target_safety_factor"]
        > result.output["sigma_yield_mpa"]
    )
    assert result.output["model"] == "von_mises_first_order"


def test_fea_analyze_passes_for_light_load():
    fea = FEAStressAgent()
    result = _analyze(fea, _BRACKET, _LIGHT_LOAD)
    assert result.ok
    assert result.output["passed"]
    assert result.output["factor_of_safety"] > 2.0


def test_fea_reinforce_grows_cross_section_and_passes_reevaluation():
    fea = FEAStressAgent()
    before = _analyze(fea, _BRACKET, _HEAVY_LOAD).output
    result = fea.process(
        AgentRequest(
            "fea_stress",
            "reinforce",
            {
                "object": _BRACKET,
                "target_safety_factor": 2.0,
                "current_safety_factor": before["factor_of_safety"],
            },
        )
    )
    assert result.ok
    parameters = result.output["parameters"]
    assert parameters["width"] > 30 and parameters["height"] > 50
    reinforced = {**_BRACKET, "parameters": parameters}
    after = _analyze(fea, reinforced, _HEAVY_LOAD)
    assert after.ok
    assert after.output["factor_of_safety"] >= 2.0


def test_fea_reinforce_noop_when_already_safe():
    fea = FEAStressAgent()
    before = _analyze(fea, _BRACKET, _LIGHT_LOAD).output
    result = fea.process(
        AgentRequest(
            "fea_stress",
            "reinforce",
            {
                "object": _BRACKET,
                "target_safety_factor": 2.0,
                "current_safety_factor": before["factor_of_safety"],
            },
        )
    )
    assert result.ok
    assert result.output["reinforced"] is False
    assert result.output["parameters"] == _BRACKET["parameters"]


def test_reinforcement_policy_formula():
    policy = ReinforcementPolicy()
    growth = policy.required_growth(current_sf=1.25, target_sf=2.0)
    assert growth == pytest.approx(1.264911, rel=1e-4)
    with pytest.raises(ValueError):
        ReinforcementPolicy(max_growth_per_step=1.0)
    with pytest.raises(ValueError):
        policy.required_growth(1.0, 0.0)


# ------------------------------------------------------------------ DFM agent


def test_dfm_assess_flags_thin_wall():
    agent = DFMManufacturingAgent()
    result = agent.process(
        AgentRequest(
            "dfm_manufacturing",
            "assess",
            {"part": {"min_wall_thickness": 0.5}, "processes": ["machining"]},
        )
    )
    assert not result.ok
    assert "machining_min_wall" in result.output["errors"]
    assert result.output["recommendations"]


def test_dfm_recommend_process_picks_viable_alternative():
    agent = DFMManufacturingAgent()
    result = agent.process(
        AgentRequest(
            "dfm_manufacturing",
            "recommend_process",
            {"part": {"min_wall_thickness": 0.5}},
        )
    )
    assert result.ok
    recommended = result.output["recommended"]
    by_name = {row["process"]: row for row in result.output["scores"]}
    assert by_name["machining"]["viable"] is False
    assert recommended is not None
    assert recommended != "machining"
    assert by_name[recommended]["viable"] is True


# ----------------------------------------------------------- cost estimator


def test_cost_estimator_estimate():
    agent = CostEstimatorAgent()
    result = agent.process(
        AgentRequest(
            "cost_estimator",
            "estimate",
            {"part": _BRACKET, "process": "machining", "quantity": 100},
        )
    )
    assert result.ok
    assert result.output["mass_kg"] > 0
    assert result.output["unit_cost_usd"] > 0
    assert result.output["total_cost_usd"] > result.output["unit_cost_usd"]


def test_cost_estimator_compare_iterations():
    agent = CostEstimatorAgent()
    result = agent.process(
        AgentRequest(
            "cost_estimator",
            "compare_iterations",
            {"iterations": [{"part": _BRACKET}, {"part": _BRACKET}]},
        )
    )
    assert result.ok
    rows = result.output["rows"]
    assert len(rows) == 2
    assert rows[0]["delta_usd"] is None
    assert rows[1]["delta_usd"] == 0.0


# ------------------------------------------------------------------ the loop


def test_loop_converges_after_automatic_reinforcement():
    loop = DesignOrchestrationLoop(max_iterations=5, target_safety_factor=2.0)
    report = loop.run(
        {
            "goal": "bracket under heavy load",
            "part": _BRACKET,
            "load_cases": [_HEAVY_LOAD],
            "target_safety_factor": 2.0,
        }
    )
    assert report.converged
    assert len(report.iterations) >= 2
    assert report.final_parameters["width"] > 30
    assert report.summary["final_safety_factor"] >= 2.0
    assert report.summary["reinforcement_steps"] >= 1


def test_loop_terminates_without_convergence_when_bounded():
    loop = DesignOrchestrationLoop(max_iterations=3, target_safety_factor=2.0)
    thin = {
        "name": "thin",
        "feature": "block",
        "parameters": {"length": 40, "width": 10, "height": 10},
        "material": "steel",
    }
    report = loop.run({"goal": "thin part", "part": thin, "load_cases": [_HEAVY_LOAD]})
    assert not report.converged
    assert len(report.iterations) == 3


def test_loop_switches_process_when_dfm_fails():
    loop = DesignOrchestrationLoop(max_iterations=5, target_safety_factor=2.0)
    part = {
        "name": "molded",
        "feature": "block",
        "parameters": {"length": 40, "width": 30, "height": 50, "min_wall_thickness": 0.5},
        "material": "steel",
    }
    report = loop.run(
        {
            "goal": "thin-walled part",
            "part": part,
            "load_cases": [_LIGHT_LOAD],
            "process": "machining",
        }
    )
    assert report.converged
    assert report.final_process != "machining"


# ------------------------------------------------------- lead architect agent


def test_lead_architect_design_action():
    lead = LeadArchitectAgent()
    result = lead.process(
        AgentRequest(
            "lead_architect",
            "design",
            {
                "goal": "bracket under heavy load",
                "part": _BRACKET,
                "load_cases": [_HEAVY_LOAD],
                "target_safety_factor": 2.0,
            },
        )
    )
    assert result.ok
    assert result.output["converged"] is True
    assert result.output["final_parameters"]["width"] > 30


def test_lead_architect_iterate_action_returns_next_state():
    lead = build_design_swarm().lead
    state = {
        "part": _BRACKET,
        "load_cases": [_HEAVY_LOAD],
        "target_safety_factor": 2.0,
        "process": "machining",
        "iteration": 0,
    }
    result = lead.process(AgentRequest("lead_architect", "iterate", {"state": state}))
    assert result.ok
    assert result.output["state"]["iteration"] == 1
    assert not result.output["passed"]


# ---------------------------------------------------------------------- swarm


def test_build_design_swarm_wires_agents_and_registers():
    registry = AgentRegistry()
    swarm = build_design_swarm(max_iterations=7, target_safety_factor=2.5)
    swarm.register_all(registry)
    assert registry.get("lead_architect") is not None
    assert registry.get("fea_stress") is not None
    assert registry.get("dfm_manufacturing") is not None
    assert registry.get("cost_estimator") is not None
    assert swarm.loop.max_iterations == 7
    assert swarm.loop.target_safety_factor == 2.5
    assert swarm.lead.loop is swarm.loop


def test_design_swarm_imported_from_package_root():
    from cadgenesis.agents import FEAStressAgent as RootFEA
    from cadgenesis.agents import build_design_swarm as root_builder

    assert RootFEA is FEAStressAgent
    assert root_builder is build_design_swarm


def test_cad_config_design_loop_roundtrip(tmp_path):
    cfg = CADConfig()
    cfg.design.max_iterations = 4
    cfg.design.target_safety_factor = 3.0
    path = tmp_path / "config.json"
    cfg.save(path)
    restored = CADConfig.load(path)
    assert restored.design.max_iterations == 4
    assert restored.design.target_safety_factor == 3.0


def test_cad_config_rejects_invalid_design_settings():
    cfg = CADConfig()
    cfg.design.target_safety_factor = 1.0
    with pytest.raises(ValueError):
        cfg._validate()
