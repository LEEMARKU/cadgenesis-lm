import sys
sys.path.insert(0, 'src')

from cadgensis.research_lab.experimental_transformer import ExperimentalTransformer


def test_experimental_transformer_init():
    transformer = ExperimentalTransformer()
    assert transformer is not None