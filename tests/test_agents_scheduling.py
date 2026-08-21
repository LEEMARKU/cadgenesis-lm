"""Test agents scheduling module."""
import sys
sys.path.insert(0, 'src')


def test_agents_scheduling():
    from cadgensis.agents.scheduling import TaskNode, TaskGraph, DAGScheduler
    # Test TaskNode
    node = TaskNode(id='test1', func=lambda: None)
    assert node.id == 'test1'
    # Test TaskGraph
    graph = TaskGraph()
    assert graph is not None
    # Test DAGScheduler
    scheduler = DAGScheduler()
    assert scheduler is not None