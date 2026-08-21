"""Test CAD benchmarks module."""
import sys
sys.path.insert(0, 'src')


def test_cad_benchmarks():
    from cadgensis.cad.benchmarks import CADBenchmarks
    benchmarks = CADBenchmarks()
    assert benchmarks is not None