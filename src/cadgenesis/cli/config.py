"""
cadgenesis.cli.config
=====================
``python -m cadgenesis.cli.config`` — inspect and manage configuration.

Loads platform config from JSON/YAML/TOML + env overlay, prints effective
values (with secret redaction), validates, writes JSON/YAML output, and
supports dynamic reload of a running server via the REST API.
"""

from __future__ import annotations

import argparse
import json
import logging

from cadgenesis.platform.config import ConfigStore, load_config
from cadgenesis.platform.security import SecretManager

logger = logging.getLogger("cadgenesis.cli.config")

REDACT_KEYS = ("secret", "password", "token", "api_key", "key")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cadgenesis config", description="Inspect and manage platform configuration"
    )
    parser.add_argument("--file", "-f", nargs="+", help="config source(s) (JSON/YAML/TOML)")
    parser.add_argument(
        "--key", "-k", default=None, help="print a single key (dotted or flattened)"
    )
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument(
        "--reload-server", default=None, help="POST /api/v1/config/reload on a running server"
    )
    parser.add_argument("--validate", action="store_true", help="validate sources load and parse")
    parser.add_argument("--write", default=None, help="write effective config to a JSON/YAML file")
    return parser.parse_args(argv)


def _redact(data: dict, keys: tuple[str, ...] = REDACT_KEYS) -> dict:
    redacted: dict = {}
    for key, value in data.items():
        if any(token in key.lower() for token in keys) and isinstance(value, str):
            redacted[key] = SecretManager.REDACTED
        else:
            redacted[key] = value
    return redacted


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store: ConfigStore = load_config(args.file) if args.file else load_config()
    if args.validate:
        print(f"ok: {len(store.as_dict())} keys from {len(store.sources)} source(s)")
        return 0
    if args.key:
        value = store.get_nested(args.key, "NOT FOUND")
        if args.json:
            print(json.dumps({args.key: value}))
        else:
            print(value)
        return 0
    if args.reload_server:
        import urllib.request

        request = urllib.request.Request(
            args.reload_server + "/api/v1/config/reload",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            print(response.read().decode("utf-8"))
        return 0
    data = _redact(store.as_dict())
    if args.write:
        store.save(args.write)
        print(f"written: {args.write}")
    elif args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        for key in sorted(data):
            print(f"{key} = {data[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
