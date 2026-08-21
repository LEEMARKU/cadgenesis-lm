import sys
sys.path.insert(0, 'src')

from cadgensis.tokenizer.material_tokens import MaterialTokens


def test_material_tokens_init():
    tokens = MaterialTokens()
    assert tokens is not None