import sys
sys.path.insert(0, 'src')

from cadgensis.collaboration.reputation import Reputation


def test_reputation_init():
    reputation = Reputation()
    assert reputation is not None