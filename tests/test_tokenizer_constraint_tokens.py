import sys
sys.path.insert(0, 'src')

from cadgensis.tokenizer.constraint_tokens import ConstraintTokens


def test_constraint_tokens_init():
    tokens = ConstraintTokens()
    assert tokens is not None