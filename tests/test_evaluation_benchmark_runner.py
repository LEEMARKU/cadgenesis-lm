import sys
sys.path.insert(0, 'src')

from cadgensis.evaluation.benchmark_runner import BenchmarkRunner


def test_benchmark_runner_init():
    runner = BenchmarkRunner()
    assert runner is not None