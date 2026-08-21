"""tests/serving/test_api.py
============================
Route-level tests for the platform REST API, including the regression:
payload model classes must live at module scope so ``from __future__
import annotations`` + nested route definitions keep them resolvable as
request bodies (not query parameters).
"""

from __future__ import annotations

import pytest

from cadgenesis.platform.config import ConfigStore
from cadgenesis.serving.api import create_app


class FakeEngine:
    def greedy(self, text, max_len=64):
        return type(
            "R",
            (),
            {"tokens": ["BOX", "NUM_1"], "confidence": 0.9, "toon": "steel", "stopped_on_eos": True},
        )()


def _token(client) -> str:
    response = client.post("/api/v1/auth/token", data={"username": "admin", "password": "admin"})
    assert response.status_code == 200
    return response.json()["access_token"]


def _app(tmp_path):
    config = ConfigStore(defaults={"registry.directory": str(tmp_path / "registry")})
    return create_app(config=config, engine=FakeEngine())


class TestAuthenticatedBodyModels:
    def test_generate_body_model_parses(self, tmp_path):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        with TestClient(_app(tmp_path)) as client:
            headers = {"Authorization": "Bearer " + _token(client)}
            response = client.post(
                "/api/v1/inference/generate",
                json={"text": "create a steel box", "max_len": 4},
                headers=headers,
            )
            assert response.status_code == 200
            assert response.json()["tokens"] == ["BOX", "NUM_1"]

    def test_generate_rejects_bad_body(self, tmp_path):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        with TestClient(_app(tmp_path)) as client:
            headers = {"Authorization": "Bearer " + _token(client)}
            response = client.post("/api/v1/inference/generate", json={}, headers=headers)
            assert response.status_code == 422

    def test_generate_requires_auth(self, tmp_path):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        with TestClient(_app(tmp_path)) as client:
            response = client.post(
                "/api/v1/inference/generate", json={"text": "box", "max_len": 4}
            )
            assert response.status_code == 403

    def test_register_promote_list_roundtrip(self, tmp_path):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        with TestClient(_app(tmp_path)) as client:
            headers = {"Authorization": "Bearer " + _token(client)}
            registered = client.post(
                "/api/v1/registry/models",
                json={"name": "cadgenesis", "path": "checkpoints/x.pt"},
                headers=headers,
            )
            assert registered.status_code == 200
            version = registered.json()["version"]
            promoted = client.post(
                "/api/v1/registry/promote",
                json={"name": "cadgenesis", "version": version},
                headers=headers,
            )
            assert promoted.status_code == 200
            listed = client.get("/api/v1/registry/models/cadgenesis", headers=headers)
            assert listed.status_code == 200
            assert listed.json()[0]["version"] == version

    def test_openapi_schema_builds(self, tmp_path):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        with TestClient(_app(tmp_path)) as client:
            assert client.get("/openapi.json").status_code == 200
