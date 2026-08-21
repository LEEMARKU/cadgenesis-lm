import sys
sys.path.insert(0, 'src')

from cadgensis.config.cad_config import CADConfig


def test_cad_config_init():
    config = CADConfig()
    assert config is not None