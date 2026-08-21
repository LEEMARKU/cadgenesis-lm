"""tests/utils/test_hashing.py"""

from __future__ import annotations

from cadgenesis.utils.hashing import (
    content_hash,
    deduplicate_paths,
    fingerprint,
    md5_file,
    sha256_bytes,
    sha256_file,
    stable_hash,
    verify_artifact,
)


def test_sha256_bytes_known_vector():
    digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_bytes(b"abc") == digest


def test_sha256_file_matches_bytes(tmp_path):
    target = tmp_path / "payload.bin"
    target.write_bytes(b"cadgenesis")
    assert sha256_file(target) == sha256_bytes(b"cadgenesis")


def test_md5_file(tmp_path):
    target = tmp_path / "payload.bin"
    target.write_bytes(b"cadgenesis")
    assert md5_file(target) == __import__("hashlib").md5(b"cadgenesis").hexdigest()


def test_content_hash_stable_and_dict_order_independent():
    a = content_hash({"b": 1, "a": 2}, [1, 2, 3])
    b = content_hash({"a": 2, "b": 1}, [1, 2, 3])
    assert a == b


def test_content_hash_differs():
    assert content_hash("x") != content_hash("y")


def test_stable_hash_in_range():
    for value in ("abc", {"x": 1}, [1, 2], 42):
        h = stable_hash(value)
        assert 0 <= h <= 0x7FFFFFFFFFFFFFFF
        assert h == stable_hash(value)


def test_fingerprint_metadata(tmp_path):
    target = tmp_path / "ckpt.pt"
    target.write_bytes(b"\x01" * 100)
    fp = fingerprint(target)
    assert fp["size_bytes"] == 100
    assert fp["sha256"] == sha256_file(target)
    assert "mtime_ns" in fp


def test_verify_artifact(tmp_path):
    target = tmp_path / "data.bin"
    target.write_bytes(b"hello")
    digest = sha256_file(target)
    assert verify_artifact(target, digest)
    assert not verify_artifact(target, "0" * 64)


def test_deduplicate_paths(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    c = tmp_path / "c.txt"
    a.write_bytes(b"same-content")
    b.write_bytes(b"same-content")
    c.write_bytes(b"different")
    unique = deduplicate_paths([a, b, c])
    assert len(unique) == 2
