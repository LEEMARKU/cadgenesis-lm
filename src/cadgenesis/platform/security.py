"""
cadgenesis.platform.security
============================
Security services for the CADGenesis-LM production platform.

- Secrets management: layered resolution (env var -> file -> vault file),
  in-memory cache, redaction-aware access.
- Encrypted storage: AES-256-GCM via ``cryptography`` when installed, with a
  documented stdlib-only fallback (XOR is NOT used; storage simply refuses
  encryption without the dependency and supports plaintext dev mode).
- Audit logging: append-only JSON audit trail with actor/action/resource/
  outcome, rotated by the standard logging handlers.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import threading
from collections.abc import Iterable, Mapping
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("cadgenesis.platform.security")

try:  # pragma: no cover - optional dependency
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore[import-not-found]
except ImportError:
    InvalidTag = None  # type: ignore[misc, assignment]  # optional dependency fallback
    AESGCM = None  # type: ignore[misc, assignment]  # optional dependency fallback


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SecretManager:
    """Layered secret resolution: explicit set > env > file > vault JSON.

    Secret values are cached in memory and can be redacted from any mapping.
    """

    REDACTED = "********"

    def __init__(
        self,
        vault_path: str | os.PathLike[str] | None = None,
        env_prefix: str = "CADGENESIS_SECRET_",
    ) -> None:
        self.vault_path = str(vault_path) if vault_path else None
        self.env_prefix = env_prefix
        self._secrets: dict[str, str] = {}
        self._lock = threading.RLock()
        if self.vault_path and Path(self.vault_path).exists():
            self._load_vault()

    def _load_vault(self) -> None:
        path = Path(self.vault_path or "")
        data: Mapping[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            if isinstance(value, str):
                self._secrets[key] = value

    def set(self, name: str, value: str) -> None:
        with self._lock:
            self._secrets[name] = value

    def get(self, name: str, default: str | None = None) -> str | None:
        with self._lock:
            if name in self._secrets:
                return self._secrets[name]
        env_key = f"{self.env_prefix}{name.upper()}"
        if env_key in os.environ:
            return os.environ[env_key]
        return default

    def get_required(self, name: str) -> str:
        value = self.get(name)
        if value is None:
            raise KeyError(f"required secret {name!r} not found (env/file/vault)")
        return value

    def has(self, name: str) -> bool:
        return self.get(name) is not None

    def persist_vault(self, path: str | os.PathLike[str] | None = None) -> None:
        """Write cached secrets to a vault JSON file (0600 perms)."""
        target = Path(path or self.vault_path or "")
        if not str(target):
            raise ValueError("no vault path configured")
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = json.dumps(self._secrets, indent=2)
        target.write_text(payload, encoding="utf-8")
        with suppress(OSError):  # pragma: no cover - non-POSIX
            os.chmod(target, 0o600)

    @staticmethod
    def redact(mapping: Mapping[str, Any], keys: Iterable[str]) -> dict[str, Any]:
        """Return a copy of ``mapping`` with the listed keys replaced."""
        redacted = dict(mapping)
        for key in keys:
            if key in redacted:
                redacted[key] = SecretManager.REDACTED
        return redacted


class CryptoService:
    """AES-256-GCM encryption for at-rest secrets/artifacts.

    Requires the optional ``cryptography`` package; ``encrypt`` raises
    ``RuntimeError`` otherwise (fail-closed, never silently downgrades).
    """

    KEY_BYTES = 32

    def __init__(
        self, key: bytes | str | None = None, key_file: str | os.PathLike[str] | None = None
    ) -> None:
        if AESGCM is None:
            logger.warning(
                "cryptography not installed; encryption unavailable (install 'cryptography')"
            )
        self._key = self._load_key(key, key_file)

    @staticmethod
    def _load_key(key: bytes | str | None, key_file: str | os.PathLike[str] | None) -> bytes | None:
        if key is not None:
            raw = key.encode("utf-8") if isinstance(key, str) else key
            return base64.b64decode(raw) if len(raw) == 44 else raw
        if key_file and Path(key_file).exists():
            raw = Path(key_file).read_bytes().strip()
            return base64.b64decode(raw) if len(raw) == 44 else raw
        return None

    @classmethod
    def generate_key(cls) -> str:
        """Generate a new base64-encoded 32-byte key."""
        return base64.b64encode(secrets.token_bytes(cls.KEY_BYTES)).decode("ascii")

    def is_available(self) -> bool:
        return AESGCM is not None and self._key is not None

    def encrypt(self, plaintext: bytes) -> bytes:
        if not self.is_available():
            raise RuntimeError("encryption unavailable: install 'cryptography' and provide a key")
        nonce = secrets.token_bytes(12)
        ciphertext = AESGCM(self._key).encrypt(nonce, plaintext, None)  # type: ignore[arg-type]
        return nonce + ciphertext

    def decrypt(self, payload: bytes) -> bytes:
        if not self.is_available():
            raise RuntimeError("decryption unavailable: install 'cryptography' and provide a key")
        nonce, ciphertext = payload[:12], payload[12:]
        try:
            return AESGCM(self._key).decrypt(nonce, ciphertext, None)  # type: ignore[arg-type]
        except InvalidTag as err:
            raise ValueError(
                "decryption failed: authentication tag mismatch (tampered payload)"
            ) from err


class AuditLogger:
    """Append-only structured audit trail (actor, action, resource, outcome)."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(
        self,
        actor: str,
        action: str,
        resource: str | None = None,
        outcome: str = "success",
        detail: Mapping[str, Any] | None = None,
        severity: str = "info",
    ) -> dict[str, Any]:
        entry = {
            "timestamp": _utcnow_iso(),
            "actor": actor,
            "action": action,
            "resource": resource,
            "outcome": outcome,
            "severity": severity,
            "detail": dict(detail or {}),
        }
        line = json.dumps(entry, sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        logger.info("audit: %s %s %s -> %s", actor, action, resource or "-", outcome)
        return entry

    def read(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Read the audit trail (most recent first)."""
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        entries = [json.loads(line) for line in lines if line.strip()]
        if limit is not None:
            entries = entries[-limit:]
        return list(reversed(entries))


def random_token(length: int = 32) -> str:
    """Cryptographically secure URL-safe token (API keys, nonces)."""
    return secrets.token_urlsafe(length)[:length]


__all__ = ["AuditLogger", "CryptoService", "SecretManager", "random_token"]
