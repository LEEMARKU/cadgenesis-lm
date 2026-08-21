"""tests/transformer/test_dynamic_routing.py
============================================
Unit tests for dynamic computation routing (Pillar 1): computation budgeting
and early exit.
"""

from __future__ import annotations

import pytest

from cadgenesis.transformer.dynamic_routing import (
    ComputationBudget,
    DynamicRoutingController,
    EarlyExitGate,
)


class TestComputationBudget:
    def test_full_budget(self):
        budget = ComputationBudget(1.0)
        assert budget.max_layers(12) == 12

    def test_half_budget_ceil(self):
        budget = ComputationBudget(0.4)
        assert budget.max_layers(5) == 2

    def test_minimum_one(self):
        budget = ComputationBudget(0.001)
        assert budget.max_layers(4) == 1

    def test_invalid(self):
        with pytest.raises(ValueError):
            ComputationBudget(1.5)
        with pytest.raises(ValueError):
            ComputationBudget(-0.1)

    def test_report(self):
        assert ComputationBudget(0.5).report()["budget"] == 0.5


class TestEarlyExitGate:
    def test_disabled_by_default(self):
        gate = EarlyExitGate(0.0)
        assert not gate.enabled
        assert not gate.should_exit(3, 0.99, budget_cap=8)

    def test_enabled_threshold(self):
        gate = EarlyExitGate(0.9)
        assert gate.enabled
        # min_steps=2 forbids exiting before the second layer has run.
        assert not gate.should_exit(0, 0.95, budget_cap=8, min_steps=2)
        assert gate.should_exit(2, 0.95, budget_cap=8)

    def test_low_confidence_no_exit(self):
        gate = EarlyExitGate(0.9)
        assert not gate.should_exit(2, 0.3, budget_cap=8)

    def test_budget_overrides(self):
        gate = EarlyExitGate(0.0)  # disabled
        assert gate.should_exit(7, 0.1, budget_cap=8)  # cap-1 reached

    def test_min_steps(self):
        gate = EarlyExitGate(0.9)
        assert not gate.should_exit(1, 0.99, budget_cap=10, min_steps=3)
        assert gate.should_exit(3, 0.99, budget_cap=10, min_steps=3)

    def test_invalid(self):
        with pytest.raises(ValueError):
            EarlyExitGate(1.5)


class TestDynamicRoutingController:
    def test_budget_only(self):
        ctrl = DynamicRoutingController(total_layers=8, budget=0.5)
        assert ctrl.max_layers == 4
        assert ctrl.should_stop(3, confidence=None)
        assert ctrl.report()["exit_reason"] == "budget"
        assert ctrl.report()["layers_executed"] == 4

    def test_full_budget_no_early_stop_before_last(self):
        ctrl = DynamicRoutingController(total_layers=4, budget=1.0)
        assert not ctrl.should_stop(0, confidence=None)
        assert not ctrl.should_stop(1, confidence=None)
        assert not ctrl.should_stop(2, confidence=None)
        assert ctrl.should_stop(3, confidence=None)  # final layer of 4

    def test_early_exit_path(self):
        ctrl = DynamicRoutingController(total_layers=8, budget=1.0, early_exit_threshold=0.9)
        assert not ctrl.should_stop(0, confidence=0.3)
        assert ctrl.should_stop(2, confidence=0.95)
        report = ctrl.report()
        assert report["exit_reason"] == "early_exit"
        assert report["savings_fraction"] == pytest.approx(1 - 3 / 8)

    def test_done_flag(self):
        ctrl = DynamicRoutingController(total_layers=4, budget=1.0)
        assert ctrl.should_stop(0, done=True)
        assert ctrl.report()["exit_reason"] == "done"

    def test_reset(self):
        ctrl = DynamicRoutingController(total_layers=4, budget=1.0)
        ctrl.should_stop(3)
        assert ctrl.report()["exit_layer"] == 3
        ctrl.reset()
        assert ctrl.report()["exit_layer"] is None

    def test_validation(self):
        with pytest.raises(ValueError):
            DynamicRoutingController(total_layers=0)
        with pytest.raises(ValueError):
            DynamicRoutingController(total_layers=4, min_steps=0)
        with pytest.raises(ValueError):
            DynamicRoutingController(total_layers=4, min_steps=9)
