"""cadgenesis.memory.persistence
===============================
Memory pool persistence / load (v6.0, Pillar 6).

v1 (kept): a single JSON document per store, atomically written via a temp
file + rename, rebuilt exactly with ``MemoryStore.from_dict``.

v2 (added): versioned payloads that still read v1 files transparently, plus
system snapshot / rollback, an incremental append log and a file-lock so
concurrent writers on the same directory do not corrupt state.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Self

from cadgenesis.memory.memory_common import MemoryStore

_FORMAT = "cadgenesis-memory"
_VERSION_V1 = 1
_VERSION_V2 = 2
_APPEND_EXT = ".log"


class _FileLock:
    """Best-effort cross-process lock via exclusive file creation."""

    def __init__(self, lock_path: Path, timeout: float = 10.0):
        self.lock_path = lock_path
        self.timeout = timeout
        self._acquired = False

    def __enter__(self) -> Self:
        deadline = time.time() + self.timeout
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:  # noqa: PERF203 — retry-loop lock acquisition
                if time.time() > deadline:
                    raise TimeoutError(f"timed out waiting for lock {self.lock_path}") from None
                time.sleep(0.02)
            else:
                os.close(fd)
                self._acquired = True
                return self

    def __exit__(self, *exc: object) -> None:
        if self._acquired:
            with contextlib.suppress(OSError):
                os.unlink(self.lock_path)
            self._acquired = False


class MemoryPersistence:
    """JSON persistence for semantic memory pools (v1 + v2)."""

    FORMAT = _FORMAT
    VERSION = _VERSION_V2

    # ------------------------------------------------------------- encoding

    @staticmethod
    def dumps(store: MemoryStore, version: int | None = None) -> str:
        """Serialize a store to a JSON string (v2 payload by default)."""
        selected = _VERSION_V1 if version == _VERSION_V1 else _VERSION_V2
        payload: dict[str, Any] = {
            "format": _FORMAT,
            "version": selected,
            "store": store.to_dict(),
        }
        if selected == _VERSION_V2:
            payload["written_at"] = time.time()
            payload["schema"] = "v2"
        return json.dumps(payload, indent=2, sort_keys=True)

    @staticmethod
    def loads(text: str) -> MemoryStore:
        """Rebuild a store from :meth:`dumps` output (reads v1 or v2)."""
        payload = json.loads(text)
        if payload.get("format") != _FORMAT:
            raise ValueError(f"unexpected persistence format {payload.get('format')!r}")
        store_data = payload.get("store")
        if isinstance(store_data, list):
            # v2 system document: pick the named store, else the first entry.
            store_data = store_data[0] if store_data else {}
        return MemoryStore.from_dict(store_data)

    # ------------------------------------------------------------ file I/O

    @staticmethod
    def save(store: MemoryStore, path: str | Path) -> None:
        """Atomically write a store to ``path``."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        lock = _FileLock(target.with_suffix(target.suffix + ".lock"))
        with lock:
            fd, tmp = tempfile.mkstemp(
                prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(MemoryPersistence.dumps(store))
                os.replace(tmp, target)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise

    @staticmethod
    def load(path: str | Path) -> MemoryStore:
        """Read and rebuild a store from ``path``."""
        return MemoryPersistence.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def save_many(
        stores: list[MemoryStore],
        directory: str | Path,
    ) -> list[Path]:
        """Save every store as ``<directory>/<name>.json``. Returns paths."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for store in stores:
            path = root / f"{store.name}.json"
            MemoryPersistence.save(store, path)
            written.append(path)
        return written

    @staticmethod
    def load_many(
        directory: str | Path,
        names: list[str] | None = None,
    ) -> dict[str, MemoryStore]:
        """Load stores from ``<directory>/*.json`` (optionally by name)."""
        root = Path(directory)
        patterns = [f"{name}.json" for name in names] if names else ["*.json"]
        result: dict[str, MemoryStore] = {}
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                store = MemoryPersistence.load(path)
                result[store.name] = store
        return result

    # ------------------------------------------------------- v2: snapshots

    @staticmethod
    def save_system(
        stores: list[MemoryStore],
        directory: str | Path,
        label: str = "snapshot",
    ) -> Path:
        """Write one versioned system snapshot covering every store."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"system-{label}-{int(time.time())}.json"
        payload = {
            "format": _FORMAT,
            "version": _VERSION_V2,
            "written_at": time.time(),
            "schema": "v2-system",
            "label": label,
            "store": [store.to_dict() for store in stores],
        }
        lock = _FileLock(path.with_suffix(path.suffix + ".lock"))
        with lock:
            fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(root))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, indent=2, sort_keys=True))
                os.replace(tmp, path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
        return path

    @staticmethod
    def load_system(
        path: str | Path,
    ) -> dict[str, MemoryStore]:
        """Restore every store from a system snapshot document."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != _FORMAT:
            raise ValueError(f"unexpected persistence format {payload.get('format')!r}")
        raw_stores = payload.get("store", [])
        stores: dict[str, MemoryStore] = {}
        for raw in raw_stores:
            store = MemoryStore.from_dict(raw)
            stores[store.name] = store
        return stores

    # ---------------------------------------------------- v2: rollback

    @classmethod
    def snapshot(cls, stores: list[MemoryStore]) -> dict[str, dict]:
        """In-memory immutable copy of the given stores (for rollback)."""
        return {store.name: store.to_dict() for store in stores}

    @classmethod
    def rollback(
        cls,
        stores: list[MemoryStore],
        snapshot: dict[str, dict],
    ) -> list[str]:
        """Restore stores to a previously taken :meth:`snapshot`.

        Returns the names of the stores that were restored.
        """
        restored: list[str] = []
        for store in stores:
            raw = snapshot.get(store.name)
            if raw is None:
                continue
            rebuilt = MemoryStore.from_dict(raw)
            store.clear()
            for entry in rebuilt.values():
                store.add(
                    entry.key,
                    entry.content,
                    importance=entry.importance,
                    metadata=entry.metadata,
                )
            restored.append(store.name)
        return restored

    # --------------------------------------------------- v2: append log

    @staticmethod
    def append(
        store: MemoryStore,
        key: str,
        content: object,
        directory: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Append one record to the incremental log ``<name>.log``."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        log_path = root / f"{store.name}{_APPEND_EXT}"
        entry = store.add(key, content, **kwargs)
        record = {
            "format": _FORMAT,
            "version": _VERSION_V2,
            "store": store.name,
            "entry": entry.to_dict(),
        }
        lock = _FileLock(log_path.with_suffix(log_path.suffix + ".lock"))
        with lock, log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return log_path

    @staticmethod
    def replay(
        directory: str | Path,
        store: MemoryStore,
        since_timestamp: float | None = None,
    ) -> list[str]:
        """Apply logged entries onto ``store``. Returns replayed keys."""
        log_path = Path(directory) / f"{store.name}{_APPEND_EXT}"
        if not log_path.exists():
            return []
        replayed: list[str] = []
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                entry = record["entry"]
                if since_timestamp is not None and entry["created_at"] <= since_timestamp:
                    continue
                store.add(
                    entry["key"],
                    entry["content"],
                    importance=entry["importance"],
                    metadata=entry["metadata"],
                )
                replayed.append(entry["key"])
        return replayed

    @staticmethod
    def truncate_log(directory: str | Path, store: MemoryStore) -> bool:
        """Remove the incremental log for a store."""
        log_path = Path(directory) / f"{store.name}{_APPEND_EXT}"
        with contextlib.suppress(FileNotFoundError):
            os.unlink(log_path)
            return True
        return False


__all__ = ["MemoryPersistence"]
