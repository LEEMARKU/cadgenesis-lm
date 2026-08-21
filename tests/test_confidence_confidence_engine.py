import sys
sys.path.insert(0, 'src')

from cadgensis.confidence.confidence_engine import ConfidenceEngine


def test_confidence_engine_init():
    engine = ConfidenceEngine()
    assert engine is not None