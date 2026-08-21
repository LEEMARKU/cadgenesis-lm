from __future__ import annotations

import pytest

from cadgenesis.serving.lifecycle import ModelLifecycle


class FakeModel:
    def generate(self, text):
        return f"generated for {text}"


def load_fake(path, **kwargs):
    return FakeModel()


class TestModelLifecycle:
    def test_load_and_get(self, tmp_path):
        lifecycle = ModelLifecycle(load_fn=load_fake)
        served = lifecycle.load("cad", str(tmp_path / "model.pt"), version="v1")
        assert served.name == "cad"
        assert served.version == "v1"
        assert lifecycle.get("cad") is served
        assert "cad" in lifecycle.names()

    def test_load_idempotent(self, tmp_path):
        lifecycle = ModelLifecycle(load_fn=load_fake)
        first = lifecycle.load("cad", str(tmp_path / "model.pt"))
        second = lifecycle.load("cad", str(tmp_path / "model.pt"))
        assert first is second

    def test_load_failure_marks_unhealthy(self, tmp_path):
        def failing_load(path, **kwargs):
            raise RuntimeError("corrupt checkpoint")

        lifecycle = ModelLifecycle(load_fn=failing_load)
        with pytest.raises(RuntimeError):
            lifecycle.load("cad", str(tmp_path / "bad.pt"))
        record = lifecycle.get("cad")
        assert record is not None
        assert record.healthy is False
        assert "corrupt checkpoint" in record.last_error

    def test_unload(self, tmp_path):
        lifecycle = ModelLifecycle(load_fn=load_fake)
        lifecycle.load("cad", str(tmp_path / "model.pt"))
        lifecycle.unload("cad")
        assert lifecycle.get("cad") is None

    def test_engine_raises_key_error(self, tmp_path):
        lifecycle = ModelLifecycle(load_fn=load_fake)
        with pytest.raises(KeyError):
            lifecycle.engine("missing")

    def test_engine_returns_model(self, tmp_path):
        lifecycle = ModelLifecycle(load_fn=load_fake)
        lifecycle.load("cad", str(tmp_path / "model.pt"))
        assert lifecycle.engine("cad").generate("x") == "generated for x"

    def test_status(self, tmp_path):
        lifecycle = ModelLifecycle(load_fn=load_fake)
        lifecycle.load("cad", str(tmp_path / "model.pt"), version="v1")
        status = lifecycle.status()
        assert len(status) == 1
        assert status[0]["name"] == "cad"
        assert status[0]["version"] == "v1"
        assert status[0]["healthy"] is True

    def test_force_reload(self, tmp_path):
        lifecycle = ModelLifecycle(load_fn=load_fake)
        lifecycle.load("cad", str(tmp_path / "model.pt"), version="v1")
        reloaded = lifecycle.load("cad", str(tmp_path / "model.pt"), version="v2", force=True)
        assert reloaded.version == "v2"
