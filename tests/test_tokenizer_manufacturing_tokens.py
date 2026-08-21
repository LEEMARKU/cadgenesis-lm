import sys
sys.path.insert(0, 'src')

from cadgensis.tokenizer.manufacturing_tokens import ManufacturingTokens


def test_manufacturing_tokens_init():
    tokens = ManufacturingTokens()
    assert tokens is not None