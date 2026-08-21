import sys
sys.path.insert(0, 'src')

from cadgensis.ir.toon_validation import TOONValidation


def test_toon_validation_init():
    validator = TOONValidation()
    assert validator is not None