import sys
sys.path.insert(0, 'src')

from cadgensis.tokenizer.token_evolution import TokenEvolution


def test_token_evolution_init():
    evolution = TokenEvolution()
    assert evolution is not None