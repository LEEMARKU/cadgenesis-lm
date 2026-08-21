import sys
sys.path.insert(0, 'src')

from cadgensis.training.deepspeed import DeepSpeedTraining


def test_deepspeed_init():
    training = DeepSpeedTraining()
    assert training is not None