import sys
sys.path.insert(0, 'src')

from cadgensis.transformer.losses import CADLosses


def test_losses_init():
    losses = CADLosses()
    assert losses is not None