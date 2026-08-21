import sys
sys.path.insert(0, 'src')

from cadgensis.inference.mcts import MCTSNode


def test_mcts_node_init():
    node = MCTSNode(state=None, parent=None)
    assert node is not None