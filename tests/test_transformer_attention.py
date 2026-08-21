import sys
sys.path.insert(0, 'src')

from cadgensis.transformer.attention import CADAttention


def test_cad_attention_init():
    attention = CADAttention()
    assert attention is not None