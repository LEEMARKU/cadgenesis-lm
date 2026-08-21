"""Test CAD integration execution bridge module."""
import sys
sys.path.insert(0, 'src')


def test_cad_integration_exec():
    from cadgensis.cad.integration.execution_bridge import ExecutionBridge
    bridge = ExecutionBridge()
    assert bridge is not None