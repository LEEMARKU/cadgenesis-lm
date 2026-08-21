import sys
sys.path.insert(0, 'src')

from cadgensis.trust.ledger import Ledger


def test_ledger_init():
    ledger = Ledger()
    assert ledger is not None