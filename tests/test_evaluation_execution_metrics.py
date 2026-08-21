import sys
sys.path.insert(0, 'src')

from cadgensis.evaluation.execution_metrics import ExecutionMetrics


def test_execution_metrics_init():
    metrics = ExecutionMetrics()
    assert metrics is not None