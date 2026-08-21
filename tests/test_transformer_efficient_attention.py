import sys
sys.path.insert(0, 'src')

from cadgensis.transformer.efficient_attention efficient_attention


def test_efficient_attention_init():
    attention = efficient_attention()
    assert attention is not None