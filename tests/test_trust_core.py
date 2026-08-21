import sys
sys.path.insert(0, 'src')

from cadgensis.trust.core import TrustCore


def test_trust_core_init():
    trust = TrustCore()
    assert trust is not None