"""
cadgenesis.cli.serve
====================
``python -m cadgenesis.cli.serve`` — start the platform serving stack.

Options: REST (uvicorn + FastAPI), optional gRPC port, host/port, config
file, model path.  Requires the ``serve`` extra (fastapi/uvicorn).
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import suppress


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cadgenesis serve", description="Serve the CADGenesis platform API"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--grpc-port", type=int, default=0, help="also start gRPC on this port (0 = disabled)"
    )
    parser.add_argument("--model", default=None, help="checkpoint path (CADGENESIS_MODEL)")
    parser.add_argument("--config", default=None, help="platform config JSON/YAML/TOML")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "info"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    if args.model:
        os.environ["CADGENESIS_MODEL"] = args.model
    if args.config:
        os.environ["CADGENESIS_CONFIG"] = args.config

    from cadgenesis.serving.api import app

    if app is None:
        raise RuntimeError(
            "REST serving requires fastapi+uvicorn (pip install cadgenesis-lm[serve])"
        )

    import uvicorn

    if args.grpc_port:
        from cadgenesis.serving.grpc import is_available

        if is_available():
            from cadgenesis.serving.grpc import CADGenesisServicer, generate_protos

            with suppress(RuntimeError):
                generate_protos()  # stubs already generated
            from concurrent.futures import ThreadPoolExecutor

            import grpc

            from cadgenesis.serving.proto import (  # type: ignore[attr-defined]
                cadgenesis_pb2_grpc as pb2_grpc,
            )

            server = grpc.server(ThreadPoolExecutor(max_workers=args.workers * 4))
            pb2_grpc.add_CADGenesisServicer_to_server(CADGenesisServicer(engine=None), server)  # type: ignore[arg-type]
            server.add_insecure_port(f"[::]:{args.grpc_port}")
            server.start()
            print(f"gRPC listening on :{args.grpc_port}")
        else:
            print(
                "[warn] gRPC requested but grpcio/generated stubs missing;"
                f" skipping (--grpc-port {args.grpc_port})"
            )

    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
