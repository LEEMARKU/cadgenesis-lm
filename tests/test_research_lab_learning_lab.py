import sys
sys.path.insert(0, 'src')

from cadgensis.research_lab.learning_lab import LearningLab


def test_learning_lab_init():
    lab = LearningLab()
    assert lab is not None