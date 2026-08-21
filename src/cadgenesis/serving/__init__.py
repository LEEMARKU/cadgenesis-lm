"""
cadgenesis.serving
==================

Production serving stack for CADGenesis-LM v6.0 (Pillar 11).

- api: versioned REST API (FastAPI) with SSE streaming and OpenAPI
- grpc: gRPC unary/streaming/bidirectional service + proto contract
- websocket: real-time inference, progress and event streaming
- batching: dynamic batching and batch scheduling policies
- lifecycle: model loading, hot reload and health state

Optional dependencies (install with ``pip install -e .[serve]``):
``fastapi``, ``uvicorn``, ``pydantic`` for REST; ``grpcio`` (+
``grpcio-tools`` for proto compilation) for gRPC.
"""
