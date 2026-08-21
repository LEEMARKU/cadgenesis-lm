from __future__ import annotations

import pytest

from cadgenesis.platform.sdk import (
    CADGenesisSDK,
    InferenceRequest,
    InferenceResult,
    LocalBackend,
    RestBackend,
    SDKError,
)


class FakeEngine:
    def greedy(self, text, max_len=64):
        return type(
            "R",
            (),
            {"tokens": ["a", "b"], "confidence": 0.9, "toon": "steel", "stopped_on_eos": True},
        )()

    def beam(self, text, beam_width=3, max_len=64):
        return type(
            "R", (), {"tokens": ["x", "y"], "confidence": 0.8, "toon": "", "stopped_on_eos": False}
        )()


class TestLocalBackend:
    def test_generate(self):
        backend = LocalBackend(FakeEngine())
        result = backend.generate(InferenceRequest(text="hello", max_len=8))
        assert result.text == "a b"
        assert result.confidence == 0.9
        assert result.latency_ms > 0

    def test_beam(self):
        backend = LocalBackend(FakeEngine())
        result = backend.generate(InferenceRequest(text="hello", beam_width=3))
        assert result.tokens == ["x", "y"]


class TestRestBackend:
    def test_connection_error(self):
        backend = RestBackend("http://127.0.0.1:1", timeout=0.5)
        with pytest.raises(SDKError):
            backend.generate(InferenceRequest(text="x"))

    def test_health_unreachable(self):
        backend = RestBackend("http://127.0.0.1:1")
        health = backend.health()
        assert health.get("ok") is False


class TestSDK:
    def test_requires_backend(self):
        with pytest.raises(ValueError):
            CADGenesisSDK()

    def test_local_sdk(self):
        sdk = CADGenesisSDK(local_engine=FakeEngine())
        result = sdk.generate("hello", max_len=8)
        assert isinstance(result, InferenceResult)
        assert result.text == "a b"

    def test_health_local(self):
        sdk = CADGenesisSDK(local_engine=FakeEngine())
        assert sdk.health()["ok"] is True
