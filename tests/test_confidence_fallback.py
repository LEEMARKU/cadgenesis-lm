import sys
sys.path.insert(0, 'src')

from cadgensis.confidence.fallback import FallbackEngine


def test_fallback_engine_init():
    engine = FallbackEngine()
    assert engine is not None