"""Test CAD integration simulation bridge module."""
import sys
sys.path.insert(0, 'src')


def test_cad_integration_sim():
    from cadgensis.cad.integration.simulation_bridge import SimulationBridge
    bridge = SimulationBridge()
    assert bridge is not None