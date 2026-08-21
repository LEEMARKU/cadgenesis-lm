"""Test CLI diagnostics module."""
import sys
sys.path.insert(0, 'src')


def test_cli_diagnostics():
    from cadgensis.cli.diagnostics import CLIDiagnostics
    diagnostics = CLIDiagnostics()
    assert diagnostics is not None