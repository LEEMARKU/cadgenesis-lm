import sys
sys.path.insert(0, 'src')

from cadgensis.training.packing import PackingManager


def test_packing_manager_init():
    manager = PackingManager()
    assert manager is not None