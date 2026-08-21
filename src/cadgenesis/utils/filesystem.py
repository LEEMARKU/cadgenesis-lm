"""cadgenesis.utils.filesystem
===========================
Filesystem helpers for CADGenesis-LM v6.0: atomic writes, safe path
resolution, recursive traversal, locking, and human-readable sizes.
"""

from __future__ import annotations

import contextlib
import errno
import os
import shutil
import tempfile
import time
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

PathLike = str | os.PathLike


def ensure_dir(path: PathLike) -> Path:
    """Create the directory (and parents) if missing; returns it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: PathLike, content: str, encoding: str = "utf-8") -> Path:
    """Atomically write text content to ``path``.

    The payload is written to a temporary file in the same directory and
    renamed into place, so readers never observe a partially-written file.
    """
    p = Path(path)
    ensure_dir(p.parent)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(content)
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return p


def atomic_write_bytes(path: PathLike, data: bytes) -> Path:
    """Atomically write binary content to ``path`` (see :func:`atomic_write_text`)."""
    p = Path(path)
    ensure_dir(p.parent)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return p


def safe_join(root: PathLike, *parts: PathLike) -> Path:
    """Join ``parts`` under ``root`` rejecting any traversal outside ``root``.

    Raises:
        ValueError: if the resolved path escapes the root directory.
    """
    base = Path(root).resolve()
    candidate = base.joinpath(*[Path(p) for p in parts])
    resolved = candidate.resolve()
    if not (resolved == base or base in resolved.parents):
        raise ValueError(f"Path {resolved} escapes root {base}")
    return resolved


def iter_files(
    root: PathLike,
    pattern: str = "**/*",
    extensions: set[str] | None = None,
    sort: bool = True,
) -> Iterator[Path]:
    """Iterate files under ``root`` matching an optional extension filter.

    Args:
        root: Directory to scan.
        pattern: Glob pattern relative to ``root``.
        extensions: If set, only files whose suffix (lowercased, with dot) is
            in this set are yielded.
        sort: When True, yield in deterministic sorted order.
    """
    base = Path(root)
    files = [p for p in base.glob(pattern) if p.is_file()]
    if extensions is not None:
        wanted = {ext if ext.startswith(".") else f".{ext}" for ext in extensions}
        files = [p for p in files if p.suffix.lower() in wanted]
    if sort:
        files.sort(key=lambda p: str(p).lower())
    yield from files


def human_readable_size(num_bytes: float, decimals: int = 2) -> str:
    """Format a byte count as a human-readable string (e.g. ``3.50 MiB``)."""
    value = float(num_bytes)
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.{decimals}f} {unit}"
        value /= 1024.0
    return f"{value:.{decimals}f} {units[-1]}"


@contextlib.contextmanager
def temp_dir(prefix: str = "cadgenesis_") -> Generator[Path, None, None]:
    """Create a temporary directory and remove it on exit (even on errors)."""
    path = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@contextlib.contextmanager
def file_lock(
    path: PathLike,
    blocking: bool = True,
    timeout: float = 30.0,
) -> Generator[None, None, None]:
    """Cross-process advisory lock via a lockfile.

    Uses ``msvcrt`` on Windows and ``fcntl`` on POSIX.  The lock is released
    automatically on exit, including on exceptions.
    """
    lock_path = Path(path)
    ensure_dir(lock_path.parent)
    handle = lock_path.open("a+")
    try:
        if os.name == "nt":
            import msvcrt

            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if not blocking:
                        raise
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError(f"Could not acquire lock {lock_path}") from None
                    time.sleep(0.05)
            try:
                yield
            finally:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        else:  # pragma: no cover - POSIX branch
            import fcntl

            fcntl_module: Any = fcntl  # typeshed guards fcntl attrs on win32

            mode = fcntl_module.LOCK_EX if blocking else fcntl_module.LOCK_EX | fcntl_module.LOCK_NB
            if not blocking:
                try:
                    fcntl_module.flock(handle.fileno(), mode)
                except OSError as exc:
                    if exc.errno in (errno.EACCES, errno.EAGAIN):
                        raise TimeoutError(f"Could not acquire lock {lock_path}") from exc
                    raise
            else:
                fcntl_module.flock(handle.fileno(), mode)
            try:
                yield
            finally:
                fcntl_module.flock(handle.fileno(), fcntl_module.LOCK_UN)
    finally:
        handle.close()


def copy_tree(src: PathLike, dst: PathLike, ignore: list[str] | None = None) -> Path:
    """Copy a directory tree; returns the destination path."""
    src_path, dst_path = Path(src), Path(dst)
    ignore_set = set(ignore or [])
    shutil.copytree(
        src_path,
        dst_path,
        ignore=shutil.ignore_patterns(*ignore_set) if ignore_set else None,
    )
    return dst_path


def list_files_recursive(root: PathLike, extensions: set[str] | None = None) -> list[Path]:
    """Convenience wrapper returning a sorted list of files under ``root``."""
    return list(iter_files(root, extensions=extensions))
