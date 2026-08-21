"""Test agents design DFM module."""
import sys
sys.path.insert(0, 'src')


def test_agents_design_dfm():
    from cadgensis.agents.design.dfm import DFMManufacturingAgent
    agent = DFMManufacturingAgent()
    assert agent is not None