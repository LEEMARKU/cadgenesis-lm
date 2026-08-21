import sys
sys.path.insert(0, 'src')

from cadgensis.distillation.distill_pipeline import DistillPipeline


def test_distill_pipeline_init():
    pipeline = DistillPipeline()
    assert pipeline is not None