import sys
sys.path.insert(0, 'src')

from cadgensis.confidence.risk import RiskCalculator


def test_risk_calculator_init():
    calculator = RiskCalculator()
    assert calculator is not None