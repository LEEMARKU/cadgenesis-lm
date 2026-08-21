"""Test autonomous research runner module."""
import sys
sys.path.insert(0, 'src')


def test_autonomous_research_runner():
    from cadgensis.autonomous_research.runner import Runner
    runner = Runner()
    assert runner is not None