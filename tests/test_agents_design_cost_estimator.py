"""Test agents design cost estimator module."""
import sys
sys.path.insert(0, 'src')


def test_agents_design_cost_estimator():
    from cadgensis.agents.design.cost_estimator import CostEstimatorAgent
    agent = CostEstimatorAgent()
    assert agent is not None