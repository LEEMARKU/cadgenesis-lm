"""
cadgenesis.cli.diagnostics
==========================
``python -m cadgenesis.cli.diagnostics`` — platform diagnostics.

Runs health checks, environment capture, dependency audit and produces a
diagnostics bundle (JSON) suitable for issue reports.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from importlib.metadata import PackageNotFoundError, version
from typing import cast


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cadgenesis diagnostics", description="Collect platform diagnostics"
    )
    parser.add_argument("--output", default=None, help="write diagnostics JSON")
    parser.add_argument("--no-health", action="store_true")
    return parser.parse_args(argv)


def _python_env() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "arch": platform.machine(),
        "cwd": os.getcwd(),
        "cpu_count": os.cpu_count(),
    }


def _dependencies() -> dict[str, str]:
    wanted = [
        "torch",
        "fastapi",
        "uvicorn",
        "grpcio",
        "pyyaml",
        "pydantic",
        "psutil",
        "numpy",
        "cryptography",
    ]
    found: dict[str, str] = {}
    for package in wanted:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = "not installed"
    return found


def _health() -> dict:
    from cadgenesis.monitoring.health import HealthChecker, check_disk_usage, check_memory_usage

    checker = HealthChecker()
    checker.register("memory", check_memory_usage)
    checker.register("disk", lambda: check_disk_usage("."))
    return checker.summary()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bundle: dict = {
        "cadgenesis": {
            "version": __import__("cadgenesis").__version__,
            "module_path": os.path.dirname(cast(str, __import__("cadgenesis").__file__)),
        },
        "environment": _python_env(),
        "dependencies": _dependencies(),
        "env_redacted": {
            key: "***"
            if any(t in key.lower() for t in ("key", "secret", "password", "token"))
            else value
            for key, value in sorted(os.environ.items())
            if key.startswith("CADGENESIS")
        },
    }
    if not args.no_health:
        bundle["health"] = _health()
    text = json.dumps(bundle, indent=2, default=str)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"diagnostics written: {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
