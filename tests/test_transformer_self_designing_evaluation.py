import sys
sys.path.insert(0, 'src')

from cadgensis.transformer.self_designing.evaluation import SelfDesigningEvaluation


def test_self_designing_evaluation_init():
    evaluation = SelfDesigningEvaluation()
    assert evaluation is not None