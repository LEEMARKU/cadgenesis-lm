import sys
sys.path.insert(0, 'src')

from cadgensis.execution.cost_estimation import CostEstimator


def test_cost_estimation_init():
    estimator = CostEstimator()
    assert estimator is not None