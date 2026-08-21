import sys
sys.path.insert(0, 'src')

from cadgensis.world_model.simulator import WorldModelSimulator


def test_simulator_init():
    simulator = WorldModelSimulator()
    assert simulator is not None