"""Test autonomous platform orchestrator module."""
import sys
sys.path.insert(0, 'src')


def test_autonomous_platform_orth():
    from cadgensis.autonomous_platform.orchestrator import PlatformOrchestrator
    orch = PlatformOrchestrator()
    assert orch is not None