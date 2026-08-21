"""Test agents design loop module."""
import sys
sys.path.insert(0, 'src')


def test_agents_design_loop():
    from cadgensis.agents.design.loop import DesignOrchestrationLoop
    loop = DesignOrchestrationLoop()
    assert loop is not None