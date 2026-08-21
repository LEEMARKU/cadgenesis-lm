"""cadgenesis.utils.hashing
========================
Content hashing for CADGenesis-LM v6.0: file fingerprints, checkpoint
deduplication, and stable hashing of arbitrary JSON-serializable content.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

PathLike = str | os.PathLike

_HASH_CHUNK = 1 << 20  # 1 MiB


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: PathLike, chunk_size: int = _HASH_CHUNK) -> str:
    """Streaming hex SHA-256 digest of a file's contents."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: PathLike, chunk_size: int = _HASH_CHUNK) -> str:
    """Streaming hex MD5 digest of a file's contents."""
    digest = hashlib.md5()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    """Serialise arbitrary content into a canonical byte string for hashing."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, (str, int, float, bool, type(None), list, dict, tuple)):
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    return repr(value).encode("utf-8")


def content_hash(*values: Any) -> str:
    """Stable hex SHA-256 of one or more JSON-serializable values.

    Dictionaries are sorted by key and tuples are treated as lists, so the
    digest is stable across Python runs and platforms.
    """
    digest = hashlib.sha256()
    for value in values:
        digest.update(_canonical(value))
    return digest.hexdigest()


def stable_hash(value: Any) -> int:
    """Deterministic 63-bit integer hash of a value (stable across runs).

    Useful for seeding experiments from structured configs without relying on
    ``hash()`` (which is salted per-process in CPython).
    """
    digest = hashlib.blake2b(_canonical(value), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFFFFFFFFFF


def fingerprint(path: PathLike, include_meta: bool = True) -> dict[str, Any]:
    """Compute a versionable fingerprint for an artifact file.

    Returns ``{"path", "size_bytes", "mtime_ns", "sha256"}``; when
    ``include_meta`` is False only the content digest is computed.
    """
    p = Path(path)
    stat = p.stat()
    result: dict[str, Any] = {"path": str(p), "sha256": sha256_file(p)}
    if include_meta:
        result["size_bytes"] = stat.st_size
        result["mtime_ns"] = stat.st_mtime_ns
    return result


def deduplicate_paths(paths: list[PathLike]) -> list[str]:
    """Return unique file paths based on content hashing (first occurrence wins)."""
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        digest = sha256_file(path)
        if digest not in seen:
            seen.add(digest)
            unique.append(str(path))
    return unique


def verify_artifact(path: PathLike, expected_sha256: str) -> bool:
    """Return True when the file's SHA-256 matches ``expected_sha256``."""
    try:
        return sha256_file(path) == expected_sha256
    except (OSError, ValueError):
        return False
