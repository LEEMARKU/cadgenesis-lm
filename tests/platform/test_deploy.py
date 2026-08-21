"""tests/platform/test_deploy.py
================================
CLI deployment tests: local registry path and remote REST path.
"""

from __future__ import annotations

import json

import pytest

from cadgenesis.cli import deploy
from cadgenesis.platform.sdk import RestBackend


class FakeRestBackend:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict]] = []
        self.gets: list[str] = []

    def _post(self, path: str, payload: dict) -> dict:
        self.posts.append((path, payload))
        return {"name": payload["name"], "version": "v1"}

    def _get(self, path: str) -> list[dict]:
        self.gets.append(path)
        return [{"name": "cadgenesis", "version": "v1", "path": "ckpt.pt"}]


def test_remote_list_uses_get_versions_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = FakeRestBackend()
    monkeypatch.setattr(deploy, "_remote", lambda base, key: backend)
    assert deploy.main(["--server", "http://localhost:8000", "list", "cadgenesis"]) == 0
    assert backend.gets == ["/api/v1/registry/models/cadgenesis"]
    assert backend.posts == []


def test_remote_register_promote_rollback_use_post(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = FakeRestBackend()
    monkeypatch.setattr(deploy, "_remote", lambda base, key: backend)
    assert deploy.main(["--server", "http://localhost:8000", "register", "m", "ckpt.pt"]) == 0
    assert deploy.main(
        ["--server", "http://localhost:8000", "promote", "m", "v1", "--environment", "prod"]
    ) == 0
    assert deploy.main(
        ["--server", "http://localhost:8000", "rollback", "m", "--environment", "prod"]
    ) == 0
    paths = [p for p, _ in backend.posts]
    assert paths == [
        "/api/v1/registry/models",
        "/api/v1/registry/promote",
        "/api/v1/registry/rollback",
    ]


def test_rest_backend_get(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> object:
        class Response:
            def read(self) -> bytes:
                return b'{"name": "cadgenesis", "versions": ["v1"]}'

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *args: object) -> None:
                return None

        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    backend = RestBackend("http://localhost:8000")
    assert backend._get("/api/v1/registry/models/cadgenesis")["versions"] == ["v1"]


def test_local_list_empty_registry(tmp_path) -> None:
    registry = str(tmp_path / "registry")
    assert deploy.main(["--registry", registry, "list", "cadgenesis"]) == 0


def test_local_register_and_list_roundtrip(tmp_path) -> None:
    registry = str(tmp_path / "registry")
    assert deploy.main(["--registry", registry, "register", "cadgenesis", "ckpt.pt"]) == 0
    assert deploy.main(["--registry", registry, "list", "cadgenesis"]) == 0


def test_remote_list_connection_error(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    class BrokenBackend:
        def _get(self, path: str) -> list[dict]:
            raise Exception("boom")

    monkeypatch.setattr(deploy, "_remote", lambda base, key: BrokenBackend())
    with pytest.raises(Exception, match="boom"):
        deploy.main(["--server", "http://localhost:8000", "list", "cadgenesis"])


def test_json_output_flag(tmp_path, capsys) -> None:
    registry = str(tmp_path / "registry")
    deploy.main(["--registry", registry, "register", "cadgenesis", "ckpt.pt"])
    capsys.readouterr()
    deploy.main(["--registry", registry, "--json", "list", "cadgenesis"])
    out = json.loads(capsys.readouterr().out)
    assert out[0]["name"] == "cadgenesis"