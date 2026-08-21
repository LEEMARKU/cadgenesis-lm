"""
cadgenesis.cli.deploy
=====================
``python -m cadgenesis.cli.deploy`` — deploy models via the registry.

Registers a checkpoint, promotes it to an environment, lists versions,
or rolls back.  Supports local registries and remote platform registries.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

logger = logging.getLogger("cadgenesis.cli.deploy")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cadgenesis deploy", description="Model registry deployment operations"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register", help="register a checkpoint")
    reg.add_argument("name")
    reg.add_argument("path")
    reg.add_argument("--version", default=None)
    reg.add_argument("--metadata", default=None, help="JSON metadata string")

    prom = sub.add_parser("promote", help="promote a version to an environment")
    prom.add_argument("name")
    prom.add_argument("version")
    prom.add_argument("--environment", default="production")

    roll = sub.add_parser("rollback", help="roll back an environment")
    roll.add_argument("name")
    roll.add_argument("--environment", default="production")

    ls = sub.add_parser("list", help="list registered versions")
    ls.add_argument("name")

    parser.add_argument(
        "--registry", default=None, help="registry directory (default: outputs/registry)"
    )
    parser.add_argument(
        "--server", default=None, help="remote platform API base URL (e.g. http://localhost:8000)"
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _remote(base: str, api_key: str | None):
    from cadgenesis.platform.sdk import RestBackend

    return RestBackend(base, api_key)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from cadgenesis.platform.registry import ModelRegistry

    registry_dir = args.registry or os.environ.get("CADGENESIS_REGISTRY", "outputs/registry")
    if args.server:
        backend = _remote(args.server, args.api_key)
    else:
        backend = ModelRegistry(registry_dir)

    if args.command == "register":
        metadata = json.loads(args.metadata) if args.metadata else {}
        if isinstance(backend, ModelRegistry):
            record = backend.register(args.name, args.path, version=args.version, metadata=metadata)
        else:
            record = backend._post(
                "/api/v1/registry/models",
                {
                    "name": args.name,
                    "path": args.path,
                    "version": args.version,
                    "metadata": metadata,
                },
            )
            print(
                json.dumps(record, indent=2)
                if args.json
                else f"registered {args.name}@{record['version']}"
            )
            return 0
        print(
            json.dumps(record.to_dict(), indent=2)
            if args.json
            else f"registered {record.name}@{record.version} -> {record.path}"
        )
    elif args.command == "promote":
        if isinstance(backend, ModelRegistry):
            record = backend.promote(args.name, args.version, args.environment)
            print(f"promoted {record.name}@{record.version} -> {args.environment}")
        else:
            backend._post(
                "/api/v1/registry/promote",
                {"name": args.name, "version": args.version, "environment": args.environment},
            )
            print(f"promoted {args.name}@{args.version} -> {args.environment}")
    elif args.command == "rollback":
        if isinstance(backend, ModelRegistry):
            rolled_back = backend.rollback(args.name, args.environment)
            if rolled_back is None:
                print("nothing to roll back")
            else:
                print(f"rolled back to {rolled_back.name}@{rolled_back.version}")
        else:
            backend._post(
                "/api/v1/registry/rollback", {"name": args.name, "environment": args.environment}
            )
            print(f"rolled back {args.environment}")
    elif args.command == "list":
        if isinstance(backend, ModelRegistry):
            versions = [v.to_dict() for v in backend.list_versions(args.name)]
        else:
            versions = backend._get(f"/api/v1/registry/models/{args.name}")
        print(json.dumps(versions, indent=2) if args.json else json.dumps(versions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
