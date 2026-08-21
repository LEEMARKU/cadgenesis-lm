"""Tests for Pillar 8 analytic simulation."""

from __future__ import annotations

import pytest

from cadgenesis.execution import SimulationEngine, SimulationResult

STEEL = {
    "name": "steel",
    "yield_strength_pa": 250e6,
    "density_kg_m3": 7800.0,
    "thermal_conductivity_w_mk": 50.0,
}


@pytest.fixture
def sim() -> SimulationEngine:
    return SimulationEngine()


class TestStructural:
    def test_factor_of_safety(self, sim: SimulationEngine) -> None:
        result = sim.structural(
            {
                "material": STEEL,
                "load": {"force_n": 1000.0, "area_m2": 1e-3},
            }
        )
        assert result.analysis_type == "structural"
        assert result.passed
        assert result.values["factor_of_safety"] == pytest.approx(250.0)

    def test_overload_fails(self, sim: SimulationEngine) -> None:
        result = sim.structural(
            {
                "material": STEEL,
                "load": {"force_n": 1e9, "area_m2": 1e-3},
            }
        )
        assert not result.passed

    def test_no_load_case(self, sim: SimulationEngine) -> None:
        result = sim.structural({"material": STEEL})
        assert result.passed
        assert "no load case" in result.messages[0]

    def test_no_material_ok(self, sim: SimulationEngine) -> None:
        result = sim.structural({})
        assert result.analysis_type == "structural"


class TestThermal:
    def test_thermal_balance(self, sim: SimulationEngine) -> None:
        result = sim.thermal(
            {
                "material": STEEL,
                "load": {"heat_w": 100.0},
                "dimensions": {"thickness_m": 0.01, "area_m2": 0.01},
            }
        )
        assert result.passed
        assert "delta_t_k" in result.values

    def test_overheat_fails(self, sim: SimulationEngine) -> None:
        result = sim.thermal(
            {
                "material": STEEL,
                "load": {"heat_w": 1e9},
                "dimensions": {"thickness_m": 0.01, "area_m2": 0.01},
                "max_temperature_rise_k": 100.0,
            }
        )
        assert not result.passed


class TestTolerance:
    def test_stackup(self, sim: SimulationEngine) -> None:
        result = sim.tolerance(chain=[(10.0, 0.1), (5.0, 0.05)])
        assert result.analysis_type == "tolerance"
        assert result.values["worst_case"] == pytest.approx(0.15)
        assert result.values["nominal"] == pytest.approx(15.0)
        assert result.passed

    def test_no_chain(self, sim: SimulationEngine) -> None:
        result = sim.tolerance(chain=[])
        assert result.analysis_type == "tolerance"
        assert result.values["worst_case"] == 0.0


class TestOther:
    def test_run_dispatch(self, sim: SimulationEngine) -> None:
        result = sim.run("structural", part={"material": STEEL})
        assert isinstance(result, SimulationResult)
        assert result.summary()["analysis_type"] == "structural"

    def test_unknown_analysis_raises(self, sim: SimulationEngine) -> None:
        with pytest.raises(ValueError):
            sim.run("quantum", part={})

    def test_summary_shape(self, sim: SimulationEngine) -> None:
        summary = sim.structural(
            {"material": STEEL, "load": {"force_n": 1000.0, "area_m2": 1e-3}}
        ).summary()
        assert set(summary) == {"analysis_type", "passed", "values", "messages", "model"}
