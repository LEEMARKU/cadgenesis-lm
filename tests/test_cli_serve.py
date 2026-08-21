"""Test CLI serve module."""
import sys
sys.path.insert(0, 'src')


def test_cli_serve():
    from cadgensis.cli.serve import serve
    # Just verify the function exists
    assert callable(serve)