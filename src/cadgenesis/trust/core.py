"""Trust Layer Core - Immutable records, signatures, integrity verification."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa


class RecordType(str, Enum):
    """Types of records that can be stored in the trust layer."""

    DATASET = "dataset"
    MODEL = "model"
    CAD_ASSET = "cad_asset"
    EXPERIMENT = "experiment"
    PLUGIN = "plugin"
    ADAPTER = "adapter"
    FEDERATED_ROUND = "federated_round"


@dataclass(frozen=True)
class TrustRecord:
    """Immutable cryptographic record."""

    record_id: str
    record_type: RecordType
    payload_hash: str
    payload: dict[str, Any]
    timestamp: float
    signature: str
    signer_public_key: str
    previous_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "record_type": self.record_type.value,
            "payload_hash": self.payload_hash,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "signature": self.signature,
            "signer_public_key": self.signer_public_key,
            "previous_hash": self.previous_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrustRecord:
        return cls(
            record_id=data["record_id"],
            record_type=RecordType(data["record_type"]),
            payload_hash=data["payload_hash"],
            payload=data["payload"],
            timestamp=data["timestamp"],
            signature=data["signature"],
            signer_public_key=data["signer_public_key"],
            previous_hash=data.get("previous_hash"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TrustConfig:
    """Configuration for the trust layer."""

    enable_blockchain: bool = False
    blockchain_backend: str = "local"  # local, ethereum, hyperledger, polygon
    blockchain_config: dict[str, Any] = field(default_factory=dict)
    signing_algorithm: str = "ed25519"  # ed25519, rsa
    key_rotation_interval_days: int = 90
    max_record_size_mb: int = 10
    storage_path: str = "./trust_storage"
    enable_audit_log: bool = True
    audit_log_path: str = "./trust_audit.log"


class KeyManager:
    """Manages cryptographic keys for signing and verification."""

    def __init__(self, config: TrustConfig):
        self.config = config
        self._private_key: ed25519.Ed25519PrivateKey | rsa.RSAPrivateKey | None = None
        self._public_key: ed25519.Ed25519PublicKey | rsa.RSAPublicKey | None = None
        self._lock = threading.RLock()
        self._generate_keys()

    def _generate_keys(self) -> None:
        if self.config.signing_algorithm == "ed25519":
            self._private_key = ed25519.Ed25519PrivateKey.generate()
            self._public_key = self._private_key.public_key()
        elif self.config.signing_algorithm == "rsa":
            self._private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            self._public_key = self._private_key.public_key()

    def get_private_key(self) -> ed25519.Ed25519PrivateKey | rsa.RSAPrivateKey:
        with self._lock:
            assert self._private_key is not None
            return self._private_key

    def get_public_key(self) -> ed25519.Ed25519PublicKey | rsa.RSAPublicKey:
        with self._lock:
            assert self._public_key is not None
            return self._public_key

    def get_public_key_pem(self) -> str:
        with self._lock:
            assert self._public_key is not None
            if isinstance(self._public_key, ed25519.Ed25519PublicKey):
                return self._public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                ).decode()
            else:
                return self._public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                ).decode()

    def sign(self, data: bytes) -> str:
        with self._lock:
            assert self._private_key is not None
            if isinstance(self._private_key, ed25519.Ed25519PrivateKey):
                signature = self._private_key.sign(data)
            else:
                signature = self._private_key.sign(
                    data,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH,
                    ),
                    hashes.SHA256(),
                )
            return signature.hex()

    def verify(self, data: bytes, signature_hex: str, public_key_pem: str) -> bool:
        try:
            public_key = serialization.load_pem_public_key(public_key_pem.encode())
            signature = bytes.fromhex(signature_hex)
            if isinstance(public_key, ed25519.Ed25519PublicKey):
                public_key.verify(signature, data)
            else:
                rsa_key = cast(rsa.RSAPublicKey, public_key)  # only ed25519/rsa keys are loaded
                rsa_key.verify(
                    signature,
                    data,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH,
                    ),
                    hashes.SHA256(),
                )
            return True
        except (InvalidSignature, ValueError):
            return False


class TrustLayer:
    """Core trust layer providing immutable, cryptographically verified records."""

    def __init__(self, config: TrustConfig | None = None):
        self.config = config or TrustConfig()
        self.key_manager = KeyManager(self.config)
        self._records: list[TrustRecord] = []
        self._record_index: dict[str, TrustRecord] = {}
        self._type_index: dict[RecordType, list[str]] = {rt: [] for rt in RecordType}
        self._lock = threading.RLock()
        self._last_hash: str | None = None
        self._audit_log = None
        if self.config.enable_audit_log:
            import logging

            self._audit_log = logging.getLogger("trust_audit")
            handler = logging.FileHandler(self.config.audit_log_path)
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self._audit_log.addHandler(handler)
            self._audit_log.setLevel(logging.INFO)

    def _compute_hash(self, data: dict[str, Any]) -> str:
        """Compute SHA-256 hash of canonical JSON representation."""
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def create_record(
        self,
        record_type: RecordType,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> TrustRecord:
        """Create a new signed, immutable record."""
        with self._lock:
            payload_hash = self._compute_hash(payload)
            timestamp = time.time()
            record_id = str(uuid.uuid4())

            record_data = {
                "record_id": record_id,
                "record_type": record_type.value,
                "payload_hash": payload_hash,
                "payload": payload,
                "timestamp": timestamp,
                "previous_hash": self._last_hash,
                "metadata": metadata or {},
            }

            data_to_sign = json.dumps(record_data, sort_keys=True, separators=(",", ":")).encode()
            signature = self.key_manager.sign(data_to_sign)
            public_key_pem = self.key_manager.get_public_key_pem()

            record = TrustRecord(
                record_id=record_id,
                record_type=record_type,
                payload_hash=payload_hash,
                payload=payload,
                timestamp=timestamp,
                signature=signature,
                signer_public_key=public_key_pem,
                previous_hash=self._last_hash,
                metadata=metadata or {},
            )

            self._records.append(record)
            self._record_index[record_id] = record
            self._type_index[record_type].append(record_id)
            self._last_hash = payload_hash

            if self._audit_log:
                self._audit_log.info(f"CREATED: {record_type.value} {record_id}")

            return record

    def get_record(self, record_id: str) -> TrustRecord | None:
        """Retrieve a record by ID."""
        with self._lock:
            return self._record_index.get(record_id)

    def get_records_by_type(self, record_type: RecordType) -> list[TrustRecord]:
        """Get all records of a specific type."""
        with self._lock:
            return [self._record_index[rid] for rid in self._type_index[record_type]]

    def verify_record(self, record: TrustRecord) -> bool:
        """Verify the integrity and signature of a record."""
        record_data = {
            "record_id": record.record_id,
            "record_type": record.record_type.value,
            "payload_hash": record.payload_hash,
            "payload": record.payload,
            "timestamp": record.timestamp,
            "previous_hash": record.previous_hash,
            "metadata": record.metadata,
        }
        data_to_verify = json.dumps(record_data, sort_keys=True, separators=(",", ":")).encode()
        return self.key_manager.verify(data_to_verify, record.signature, record.signer_public_key)

    def verify_chain(self) -> tuple[bool, list[str]]:
        """Verify the entire hash chain integrity."""
        errors = []
        prev_hash = None
        for i, record in enumerate(self._records):
            if record.previous_hash != prev_hash:
                errors.append(f"Record {i} ({record.record_id}): hash chain broken")
            if record.payload_hash != self._compute_hash(record.payload):
                errors.append(f"Record {i} ({record.record_id}): payload hash mismatch")
            if not self.verify_record(record):
                errors.append(f"Record {i} ({record.record_id}): signature invalid")
            prev_hash = record.payload_hash
        return len(errors) == 0, errors

    def export_records(self) -> list[dict[str, Any]]:
        """Export all records as dictionaries."""
        with self._lock:
            return [r.to_dict() for r in self._records]

    def import_records(self, records: list[dict[str, Any]]) -> int:
        """Import records from exported format."""
        imported = 0
        with self._lock:
            for r_dict in records:
                record = TrustRecord.from_dict(r_dict)
                if self.verify_record(record):
                    self._records.append(record)
                    self._record_index[record.record_id] = record
                    self._type_index[record.record_type].append(record.record_id)
                    self._last_hash = record.payload_hash
                    imported += 1
        return imported
