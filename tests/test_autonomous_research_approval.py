"""Test autonomous research approval module."""
import sys
sys.path.insert(0, 'src')


def test_autonomous_research_approval():
    from cadgensis.autonomous_research.approval import ApprovalManager
    mgr = ApprovalManager()
    assert mgr is not None