import sys
sys.path.insert(0, 'src')

from cadgensis.research_lab.evaluation import EvaluationLab


def test_evaluation_lab_init():
    lab = EvaluationLab()
    assert lab is not None