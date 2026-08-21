"""tests/utils/test_filesystem.py"""

from __future__ import annotations

import contextlib
import os

import pytest

from cadgenesis.utils.filesystem import (
    atomic_write_bytes,
    atomic_write_text,
    ensure_dir,
    file_lock,
    human_readable_size,
    iter_files,
    list_files_recursive,
    safe_join,
    temp_dir,
)


def test_ensure_dir(tmp_path):
    target = tmp_path / "a" / "b"
    assert ensure_dir(target) == target
    assert target.is_dir()


def test_atomic_write_text(tmp_path):
    target = tmp_path / "out" / "file.txt"
    atomic_write_text(target, "hello")
    assert target.read_text(encoding="utf-8") == "hello"
    assert not list(tmp_path.rglob("*.tmp"))


def test_atomic_write_bytes(tmp_path):
    target = tmp_path / "out.bin"
    atomic_write_bytes(target, b"\x00\x01\x02")
    assert target.read_bytes() == b"\x00\x01\x02"


def test_safe_join_allows_within_root(tmp_path):
    path = safe_join(tmp_path, "sub", "file.txt")
    assert str(path).startswith(str(tmp_path))


def test_safe_join_rejects_escape(tmp_path):
    with pytest.raises(ValueError):
        safe_join(tmp_path, "..", "..", "etc", "passwd")


def test_iter_files_with_extensions(tmp_path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.py").write_text("b", encoding="utf-8")
    (tmp_path / "c.JSON").write_text("c", encoding="utf-8")
    files = list_files_recursive(tmp_path, extensions={"txt", "json"})
    assert {f.name for f in files} == {"a.txt", "c.JSON"}


def test_iter_files_sorted(tmp_path):
    for name in ("z", "a", "m"):
        (tmp_path / f"{name}.txt").write_text(name, encoding="utf-8")
    names = [f.name for f in iter_files(tmp_path, extensions={"txt"})]
    assert names == ["a.txt", "m.txt", "z.txt"]


def test_human_readable_size():
    assert human_readable_size(512) == "512 B"
    assert human_readable_size(1024) == "1.00 KiB"
    assert human_readable_size(3 * 1024 * 1024) == "3.00 MiB"


def test_temp_dir_cleaned(tmp_path):
    with temp_dir(prefix="cadgenesis_test_") as td:
        marker = td / "marker"
        marker.write_text("x", encoding="utf-8")
        assert marker.exists()
        kept = td
    assert not kept.exists()


def test_file_lock_serializes(tmp_path):
    lock_path = tmp_path / "lock.file"
    order = []

    def worker(label):
        with file_lock(lock_path):
            order.append(f"{label}:start")
            order.append(f"{label}:end")

    with file_lock(lock_path):
        pass

    import threading

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    starts = [i for i, e in enumerate(order) if e.endswith(":start")]
    assert starts == sorted(starts)


def test_file_lock_timeout(tmp_path):
    if os.name != "nt":
        pytest.skip("msvcrt locking is Windows-only")
    lock_path = tmp_path / "held.lock"
    handle = lock_path.open("a+")
    import msvcrt

    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        with pytest.raises(TimeoutError), file_lock(lock_path, blocking=True, timeout=0.1):
            pass
    finally:
        with contextlib.suppress(OSError):
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()
