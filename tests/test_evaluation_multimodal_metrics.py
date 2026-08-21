import sys
sys.path.insert(0, 'src')

from cadgensis.evaluation.multimodal_metrics import MultimodalMetrics


def test_multimodal_metrics_init():
    metrics = MultimodalMetrics()
    assert metrics is not None