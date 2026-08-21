"""Test CAD validation checks module."""
import sys
sys.path.insert(0, 'src')


def test_cad_validation_checks():
    from cadgensis.cad.validation.checks import ValidationChecks
    checks = ValidationChecks()
    assert checks is not None