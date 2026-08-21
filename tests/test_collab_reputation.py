"""Test collaboration reputation module."""
import sys
sys.path.insert(0, 'src')


def test_collab_reputation():
    from cadgensis.collaboration.reputation import Reputation
    reputation = Reputation()
    assert reputation is not None