import sys
sys.path.insert(0, 'src')

from cadgensis.tokenizer.numeric_tokens import NumericTokens


def test_numeric_tokens_init():
    tokens = NumericTokens()
    assert tokens is not None