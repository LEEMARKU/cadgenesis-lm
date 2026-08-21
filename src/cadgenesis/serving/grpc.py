"""
cadgenesis.serving.grpc
=======================
gRPC service for the CADGenesis-LM platform.

- Unary: ``Generate`` (single prompt -> result)
- Server streaming: ``StreamGenerate`` (token events)
- Bidirectional streaming: ``ChatGenerate`` (client sends prompts, server
  replies with results)

The ``.proto`` contract lives in ``serving/proto/cadgenesis.proto``.
``CADGenesisServicer`` implements the generated service (from
``cadgenesis_serving_pb2_grpc``); ``generate_protos()`` compiles the proto
with ``grpcio-tools`` when available.  When gRPC is not installed, the module
still imports and exposes ``is_available()`` so callers can degrade to REST.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger("cadgenesis.serving.grpc")

try:  # pragma: no cover - optional dependency
    import grpc  # type: ignore[import-not-found]
except ImportError:
    grpc = None  # type: ignore[assignment]

try:  # pragma: no cover - generated stubs
    from cadgenesis.serving.proto import cadgenesis_pb2 as pb2  # type: ignore[attr-defined]
    from cadgenesis.serving.proto import (  # type: ignore[attr-defined]
        cadgenesis_pb2_grpc as pb2_grpc,
    )
except ImportError:  # pragma: no cover
    pb2 = None  # type: ignore[assignment]
    pb2_grpc = None  # type: ignore[assignment]

PROTO_SOURCE = __file__.rsplit("grpc.py", 1)[0] + "proto/cadgenesis.proto"


def is_available() -> bool:
    """True when both grpc and the generated stubs are importable."""
    return grpc is not None and pb2 is not None and pb2_grpc is not None


def generate_protos(out_dir: str | None = None) -> str | None:
    """Compile the proto contract with ``grpc_tools.protoc`` (optional dep)."""
    if grpc is None:
        raise RuntimeError("gRPC support requires 'grpcio'")
    try:
        from grpc_tools import protoc  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("proto generation requires 'grpcio-tools'") from exc
    import os

    target = out_dir or os.path.dirname(PROTO_SOURCE)
    proto_root = os.path.dirname(PROTO_SOURCE)
    protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{proto_root}",
            f"--python_out={target}",
            f"--grpc_python_out={target}",
            PROTO_SOURCE,
        ]
    )
    logger.info("compiled %s -> %s", PROTO_SOURCE, target)
    return target


class CADGenesisServicer:
    """gRPC service implementation (mirrors the proto contract).

    Handlers take ``(request, context)`` like generated servicers.  The
    servicer is transport-independent: it delegates to an injected engine
    callable ``generate(text, max_len) -> GenerationResult``.
    """

    def __init__(self, engine: Any, generate: Any | None = None) -> None:
        self.engine = engine
        self._generate = generate or (lambda text, max_len=64: engine.greedy(text, max_len=max_len))

    # unary -----------------------------------------------------------------

    def Generate(self, request: Any, context: Any) -> Any:
        result = self._generate(request.text, max_len=request.max_len)
        return pb2.GenerateResponse(
            text=" ".join(result.tokens),
            tokens=list(result.tokens),
            confidence=round(float(result.confidence), 6),
            toon=result.toon,
        )

    # server streaming -------------------------------------------------------

    def StreamGenerate(self, request: Any, context: Any) -> Iterator[Any]:
        result = self._generate(request.text, max_len=request.max_len)
        for index, token in enumerate(result.tokens):
            yield pb2.StreamEvent(index=index, token=token, done=False)
        yield pb2.StreamEvent(
            index=len(result.tokens),
            token="",
            done=True,
            text=" ".join(result.tokens),
            confidence=round(float(result.confidence), 6),
        )

    # bidirectional streaming ------------------------------------------------

    def ChatGenerate(self, request_iterator: Iterator[Any], context: Any) -> Iterator[Any]:
        for request in request_iterator:
            result = self._generate(request.text, max_len=request.max_len)
            yield pb2.GenerateResponse(
                text=" ".join(result.tokens),
                tokens=list(result.tokens),
                confidence=round(float(result.confidence), 6),
                toon=result.toon,
                request_id=request.request_id,
            )

    # server ----------------------------------------------------------------

    def serve(self, port: int = 50051, max_workers: int = 10) -> None:
        """Blocking serve loop (call from the ``serve`` CLI / deployment)."""
        if not is_available():
            raise RuntimeError("gRPC serving requires 'grpcio' + generated stubs")
        server = grpc.server(
            __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=max_workers)
        )
        pb2_grpc.add_CADGenesisServicer_to_server(self, server)
        server.add_insecure_port(f"[::]:{port}")
        logger.info("gRPC server listening on :%d", port)
        server.start()
        server.wait_for_termination()


__all__ = ["PROTO_SOURCE", "CADGenesisServicer", "generate_protos", "is_available"]
