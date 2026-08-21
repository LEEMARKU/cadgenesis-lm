import sys
sys.path.insert(0, 'src')

from cadgensis.execution.topology_analysis import TopologyAnalysis


def test_topology_analysis_init():
    analysis = TopologyAnalysis()
    assert analysis is not None