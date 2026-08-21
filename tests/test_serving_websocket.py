import sys
sys.path.insert(0, 'src')

from cadgensis.serving.websocket import WebSocketServing


def test_websocket_serving_init():
    serving = WebSocketServing()
    assert serving is not None