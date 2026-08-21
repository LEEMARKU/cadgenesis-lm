"""Test CLI main module."""
import sys
sys.path.insert(0, 'src')


def test_cli_main():
    from cadgensis.cli.main import main
    # Just verify the function exists and can be called
    try:
        main()
    except SystemExit:
        pass