import sys
sys.path.insert(0, 'src')

from cadgensis.cad.integration.simulation_bridge import SimulationBridge


def test_simulation_bridge_init():
    bridge = SimulationBridge()
    assert bridge is not None