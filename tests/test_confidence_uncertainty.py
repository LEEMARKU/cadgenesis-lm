import sys
sys.path.insert(0, 'src')

from cadgensis.confidence.uncertainty import UncertaintyQuantifier


def test_uncertainty_quantifier_init():
    quantifier = UncertaintyQuantifier()
    assert quantifier is not None