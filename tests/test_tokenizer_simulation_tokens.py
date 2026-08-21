import sys
sys.path.insert(0, 'src')

from cadgensis.tokenizer.simulation_tokens import SimulationTokens


def test_simulation_tokens_init():
    tokens = SimulationTokens()
    assert tokens is not None