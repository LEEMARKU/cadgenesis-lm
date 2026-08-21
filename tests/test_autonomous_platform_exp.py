"""Test autonomous platform explainability module."""
import sys
sys.path.insert(0, 'src')


def test_autonomous_platform_exp():
    from cadgensis.autonomous_platform.explainability import ExplanationGenerator
    gen = ExplanationGenerator()
    assert gen is not None