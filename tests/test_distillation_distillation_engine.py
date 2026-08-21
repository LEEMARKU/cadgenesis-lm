import sys
sys.path.insert(0, 'src')

from cadgensis.distillation.distillation_engine import DistillationEngine


def test_distillation_engine_init():
    engine = DistillationEngine()
    assert engine is not None