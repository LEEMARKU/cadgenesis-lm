import sys
sys.path.insert(0, 'src')

from cadgensis.world_model.spatial import Spatial


def test_spatial_init():
    spatial = Spatial()
    assert spatial is not None