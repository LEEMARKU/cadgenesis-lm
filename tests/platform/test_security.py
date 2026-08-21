from __future__ import annotations

import pytest

from cadgenesis.platform.security import AuditLogger, CryptoService, SecretManager, random_token


class TestSecretManager:
    def test_set_get_has(self):
        sm = SecretManager(env_prefix="CADG_TEST_")
        sm.set("api_key", "abc")
        assert sm.has("api_key")
        assert sm.get("api_key") == "abc"
        assert sm.get_required("api_key") == "abc"

    def test_get_missing(self):
        sm = SecretManager(env_prefix="CADG_TEST_")
        assert sm.get("nope") is None
        with pytest.raises(KeyError):
            sm.get_required("nope")

    def test_redact(self):
        redacted = SecretManager.redact({"token": "secret123", "other": "visible"}, keys=["token"])
        assert redacted["token"] == SecretManager.REDACTED
        assert redacted["other"] == "visible"


class TestCryptoService:
    def test_roundtrip(self):
        svc = CryptoService(key=CryptoService.generate_key())
        if not svc.is_available():
            pytest.skip("cryptography not installed")
        ciphertext = svc.encrypt(b"secret payload")
        assert ciphertext != b"secret payload"
        assert svc.decrypt(ciphertext) == b"secret payload"

    def test_tamper_detected(self):
        svc = CryptoService(key=CryptoService.generate_key())
        if not svc.is_available():
            pytest.skip("cryptography not installed")
        ciphertext = bytearray(svc.encrypt(b"payload"))
        ciphertext[-1] ^= 0xFF
        with pytest.raises(ValueError):
            svc.decrypt(bytes(ciphertext))

    def test_unavailable_without_key(self, monkeypatch):
        svc = CryptoService()
        with pytest.raises(RuntimeError):
            svc.encrypt(b"x")


class TestAuditLogger:
    def test_append_and_read(self, tmp_path):
        logger = AuditLogger(tmp_path / "audit.jsonl")
        logger.log(actor="alice", action="train", resource="exp_1", outcome="ok")
        logger.log(actor="bob", action="deploy", resource="cad_v2", outcome="denied")
        entries = logger.read()
        assert len(entries) == 2
        assert entries[0]["actor"] == "bob"  # most recent first
        assert entries[1]["actor"] == "alice"
        assert entries[1]["action"] == "train"

    def test_read_limit(self, tmp_path):
        logger = AuditLogger(tmp_path / "audit.jsonl")
        for _ in range(5):
            logger.log(actor="a", action="x", resource="r", outcome="ok")
        assert len(logger.read(limit=2)) == 2


class TestRandomToken:
    def test_default_length(self):
        assert len(random_token()) == 32

    def test_custom_length(self):
        assert len(random_token(16)) == 16

    def test_unique(self):
        assert random_token() != random_token()
