import sys
sys.path.insert(0, 'src')

from cadgensis.transformer.constraint_attention import ConstraintAttention


def test_constraint_attention_init():
    attention = ConstraintAttention()
    assert attention is not None