import sys
sys.path.insert(0, 'src')

from cadgensis.trust.blockchain import Blockchain


def test_blockchain_init():
    blockchain = Blockchain()
    assert blockchain is not None