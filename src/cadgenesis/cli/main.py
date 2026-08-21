"""
cadgenesis.cli.main
===================
Unified CLI entry point: ``python -m cadgenesis.cli`` (or ``cadgenesis``
console script after ``pip install .``).

Commands:
    train         train the foundation model (existing)
    generate      run local generation
    serve         start the platform API (REST + optional gRPC)
    eval          evaluate checkpoints
    config        inspect/manage configuration
    benchmark     run benchmark suites
    deploy        model registry deployment operations
    diagnostics   collect platform diagnostics
"""

from __future__ import annotations

import argparse
from collections.abc import Callable


def _subcommand(name: str, module: str) -> Callable[[list[str] | None], int]:
    def run(argv: list[str] | None = None) -> int:
        import importlib

        mod = importlib.import_module(f"cadgenesis.cli.{module}")
        return int(mod.main(argv))

    run.__name__ = name
    return run


COMMANDS: dict[str, Callable[[list[str] | None], int]] = {
    "train": _subcommand("train", "train"),
    "generate": _subcommand("generate", "generate"),
    "serve": _subcommand("serve", "serve"),
    "eval": _subcommand("eval", "eval"),
    "config": _subcommand("config", "config"),
    "benchmark": _subcommand("benchmark", "benchmark"),
    "deploy": _subcommand("deploy", "deploy"),
    "diagnostics": _subcommand("diagnostics", "diagnostics"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cadgenesis", description="CADGenesis-LM platform CLI")
    parser.add_argument("command", choices=sorted(COMMANDS), help="command to run")
    args, rest = parser.parse_known_args(argv)
    return COMMANDS[args.command](rest)


if __name__ == "__main__":
    raise SystemExit(main())
