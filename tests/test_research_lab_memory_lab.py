import sys
sys.path.insert(0, 'src')

from cadgensis.research_lab.memory_lab import MemoryLab


def test_memory_lab_init():
    lab = MemoryLab()
    assert lab is not None