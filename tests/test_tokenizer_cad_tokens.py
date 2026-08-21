import sys
sys.path.insert(0, 'src')

from cadgensis.tokenizer.cad_tokens import CADTokens


def test_cad_tokens_init():
    tokens = CADTokens()
    assert tokens is not None