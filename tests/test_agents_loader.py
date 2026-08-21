"""Test agents loader module."""
import sys
sys.path.insert(0, 'src')


def test_agents_loader():
    from cadgensis.agents.loader import AgentLoader
    loader = AgentLoader()
    assert loader is not None