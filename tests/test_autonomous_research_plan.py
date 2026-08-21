"""Test autonomous research experiment planner module."""
import sys
sys.path.insert(0, 'src')


def test_autonomous_research_plan():
    from cadgensis.autonomous_research.experiment_planner import ExperimentPlanner
    planner = ExperimentPlanner()
    assert planner is not None