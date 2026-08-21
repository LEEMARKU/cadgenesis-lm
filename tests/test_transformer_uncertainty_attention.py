import sys
sys.path.insert(0, 'src')

from cadgensis.transformer.uncertainty_attention import UncertaintyAttention


def test_uncertainty_attention_init():
    attention = UncertaintyAttention()
    assert attention is not None