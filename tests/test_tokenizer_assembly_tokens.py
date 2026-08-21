import sys
sys.path.insert(0, 'src')

from cadgensis.tokenizer.assembly_tokens import AssemblyTokens


def test_assembly_tokens_init():
    tokens = AssemblyTokens()
    assert tokens is not None