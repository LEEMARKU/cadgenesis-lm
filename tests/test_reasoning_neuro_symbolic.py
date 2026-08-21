import sys
sys.path.insert(0, 'src')

from cadgensis.reasoning.neuro_symbolic import NeuroSymbolicReasoningEngine


def test_neuro_symbolic_init():
    engine = NeuroSymbolicReasoningEngine()
    assert engine is not None