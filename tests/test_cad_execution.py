"""Test CAD execution module."""
import sys
sys.path.insert(0, 'src')

from cadgensis.execution import __all__ as execution_list


def test_execution_import():
    assert execution_list is not None