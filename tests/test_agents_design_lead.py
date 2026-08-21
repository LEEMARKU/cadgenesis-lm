"""Test agents design lead module."""
import sys
sys.path.insert(0, 'src')


def test_agents_design_lead():
    from cadgensis.agents.design.lead import LeadArchitectAgent
    agent = LeadArchitectAgent()
    assert agent is not None