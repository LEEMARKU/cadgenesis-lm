import sys
sys.path.insert(0, 'src')

from cadgensis.research_lab.agent_lab import AgentLab


def test_agent_lab_init():
    lab = AgentLab()
    assert lab is not None