import sys
sys.path.insert(0, 'src')

from cadgensis.research_lab.neuro_symbolic_lab import NeuroSymbolicLab


def test_neuro_symbolic_lab_init():
    lab = NeuroSymbolicLab()
    assert lab is not None