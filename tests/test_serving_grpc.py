import sys
sys.path.insert(0, 'src')

from cadgensis.serving.grpc import GRPCServing


def test_grpc_serving_init():
    serving = GRPCServing()
    assert serving is not None