"""Registries for plugins, adapters, and secure model management."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any

from .core import RecordType, TrustLayer
from .provenance import ModelProvenance


class PluginStatus(str, Enum):
    """Plugin verification status."""

    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"


@dataclass
class PluginManifest:
    """Plugin manifest with metadata and dependencies."""

    plugin_id: str
    name: str
    version: str
    description: str
    author: str
    license: str
    homepage: str = ""
    repository: str = ""
    entry_point: str = ""
    dependencies: dict[str, str] = field(default_factory=dict)  # name -> version
    capabilities: list[str] = field(default_factory=list)
    compatibility: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    signature: str = ""
    signer_public_key: str = ""
    status: PluginStatus = PluginStatus.PENDING
    verified_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "homepage": self.homepage,
            "repository": self.repository,
            "entry_point": self.entry_point,
            "dependencies": self.dependencies,
            "capabilities": self.capabilities,
            "compatibility": self.compatibility,
            "checksum": self.checksum,
            "signature": self.signature,
            "signer_public_key": self.signer_public_key,
            "status": self.status.value,
            "verified_at": self.verified_at,
            "metadata": self.metadata,
        }


class PluginRegistry:
    """Registry for CADGenesis plugins with cryptographic verification."""

    def __init__(self, trust_layer: TrustLayer):
        self.trust_layer = trust_layer
        self._plugins: dict[str, PluginManifest] = {}
        self._lock = RLock()

    def register_plugin(
        self,
        manifest: PluginManifest,
        plugin_code: bytes,
        verify_signature: bool = True,
    ) -> tuple[bool, list[str]]:
        """Register a plugin with integrity verification."""
        errors = []

        # Verify checksum
        computed_checksum = hashlib.sha256(plugin_code).hexdigest()
        if manifest.checksum and manifest.checksum != computed_checksum:
            errors.append(
                f"Checksum mismatch: expected {manifest.checksum}, got {computed_checksum}"
            )

        # Verify signature if provided
        if (
            verify_signature
            and manifest.signature
            and manifest.signer_public_key
            and not self.trust_layer.key_manager.verify(
                plugin_code, manifest.signature, manifest.signer_public_key
            )
        ):
            errors.append("Signature verification failed")

        # Check dependencies
        for dep_name, dep_version in manifest.dependencies.items():
            if dep_name not in self._plugins:
                errors.append(f"Missing dependency: {dep_name}@{dep_version}")
            elif self._plugins[dep_name].version != dep_version:
                errors.append(
                    f"Dependency version mismatch: {dep_name}@"
                    f"{self._plugins[dep_name].version} != {dep_version}"
                )

        if errors:
            return False, errors

        manifest.status = PluginStatus.VERIFIED
        manifest.verified_at = time.time()

        with self._lock:
            self._plugins[manifest.plugin_id] = manifest

            # Create trust record
            self.trust_layer.create_record(
                RecordType.PLUGIN,
                manifest.to_dict(),
                metadata={"plugin_id": manifest.plugin_id, "action": "register"},
            )

        return True, []

    def get_plugin(self, plugin_id: str) -> PluginManifest | None:
        with self._lock:
            return self._plugins.get(plugin_id)

    def list_plugins(
        self, capability: str | None = None, status: PluginStatus | None = None
    ) -> list[PluginManifest]:
        with self._lock:
            plugins = list(self._plugins.values())
            if capability:
                plugins = [p for p in plugins if capability in p.capabilities]
            if status:
                plugins = [p for p in plugins if p.status == status]
            return plugins

    def verify_plugin(self, plugin_id: str) -> tuple[bool, list[str]]:
        with self._lock:
            manifest = self._plugins.get(plugin_id)
            if not manifest:
                return False, ["Plugin not found"]

            trust_records = self.trust_layer.get_records_by_type(RecordType.PLUGIN)
            for record in trust_records:
                if record.payload.get("plugin_id") == plugin_id:
                    valid = self.trust_layer.verify_record(record)
                    if not valid:
                        return False, ["Trust record verification failed"]
            return True, []


class AdapterType(str, Enum):
    """Types of parameter-efficient adapters."""

    LORA = "lora"
    QLORA = "qlora"
    PEFT = "peft"
    PREFIX = "prefix_tuning"
    PROMPT = "prompt_tuning"
    IA3 = "ia3"
    ADALORA = "adalora"


@dataclass
class AdapterManifest:
    """Adapter manifest with configuration and metadata."""

    adapter_id: str
    name: str
    version: str
    adapter_type: AdapterType
    base_model: str  # provenance_id of base model
    config: dict[str, Any]
    target_modules: list[str]
    rank: int | None = None
    alpha: float | None = None
    dropout: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    training_config: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    signature: str = ""
    signer_public_key: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["adapter_type"] = self.adapter_type.value
        return d


class AdapterRegistry:
    """Registry for parameter-efficient adapters (LoRA, QLoRA, PEFT, etc.)."""

    def __init__(self, trust_layer: TrustLayer):
        self.trust_layer = trust_layer
        self._adapters: dict[str, AdapterManifest] = {}
        self._version_index: dict[str, list[str]] = {}  # base_model -> adapter_ids
        self._lock = RLock()

    def register_adapter(
        self,
        manifest: AdapterManifest,
        adapter_weights: bytes,
        verify_signature: bool = True,
    ) -> tuple[bool, list[str]]:
        errors = []

        # Verify checksum
        computed_checksum = hashlib.sha256(adapter_weights).hexdigest()
        if manifest.checksum and manifest.checksum != computed_checksum:
            errors.append(
                f"Checksum mismatch: expected {manifest.checksum}, got {computed_checksum}"
            )

        # Verify signature
        if (
            verify_signature
            and manifest.signature
            and manifest.signer_public_key
            and not self.trust_layer.key_manager.verify(
                adapter_weights, manifest.signature, manifest.signer_public_key
            )
        ):
            errors.append("Signature verification failed")

        if errors:
            return False, errors

        with self._lock:
            self._adapters[manifest.adapter_id] = manifest
            if manifest.base_model not in self._version_index:
                self._version_index[manifest.base_model] = []
            self._version_index[manifest.base_model].append(manifest.adapter_id)

            # Create trust record
            self.trust_layer.create_record(
                RecordType.ADAPTER,
                manifest.to_dict(),
                metadata={"adapter_id": manifest.adapter_id, "action": "register"},
            )

        return True, []

    def get_adapter(self, adapter_id: str) -> AdapterManifest | None:
        with self._lock:
            return self._adapters.get(adapter_id)

    def get_adapters_for_model(self, base_model: str) -> list[AdapterManifest]:
        with self._lock:
            adapter_ids = self._version_index.get(base_model, [])
            return [self._adapters[aid] for aid in adapter_ids if aid in self._adapters]

    def list_adapters(self, adapter_type: AdapterType | None = None) -> list[AdapterManifest]:
        with self._lock:
            adapters = list(self._adapters.values())
            if adapter_type:
                adapters = [a for a in adapters if a.adapter_type == adapter_type]
            return adapters

    def get_latest_adapter(
        self, base_model: str, adapter_type: AdapterType
    ) -> AdapterManifest | None:
        with self._lock:
            adapters = self.get_adapters_for_model(base_model)
            adapters = [a for a in adapters if a.adapter_type == adapter_type]
            if not adapters:
                return None
            return max(adapters, key=lambda a: a.created_at)


class SecureModelRegistry:
    """Secure model registry with versioning, rollback, and cryptographic verification."""

    def __init__(self, trust_layer: TrustLayer):
        self.trust_layer = trust_layer
        self._models: dict[str, ModelProvenance] = {}
        self._versions: dict[str, list[str]] = {}  # model_name -> list of version ids
        self._lock = RLock()

    def register_model(
        self,
        provenance: ModelProvenance,
        model_weights: bytes,
        verify_signature: bool = True,
    ) -> tuple[bool, list[str]]:
        errors = []

        # Verify checksum
        computed_checksum = hashlib.sha256(model_weights).hexdigest()
        if (
            provenance.metadata.get("checksum")
            and provenance.metadata["checksum"] != computed_checksum
        ):
            errors.append("Checksum mismatch")

        # Verify signature if provided
        if (
            verify_signature
            and provenance.metadata.get("signature")
            and provenance.metadata.get("signer_public_key")
            and not self.trust_layer.key_manager.verify(
                model_weights,
                provenance.metadata["signature"],
                provenance.metadata["signer_public_key"],
            )
        ):
            errors.append("Signature verification failed")

        if errors:
            return False, errors

        provenance.content_hash = computed_checksum

        with self._lock:
            self._models[provenance.provenance_id] = provenance
            if provenance.name not in self._versions:
                self._versions[provenance.name] = []
            self._versions[provenance.name].append(provenance.provenance_id)

            # Create trust record
            self.trust_layer.create_record(
                RecordType.MODEL,
                provenance.__dict__,
                metadata={"model_id": provenance.provenance_id, "action": "register"},
            )

        return True, []

    def get_model(self, model_id: str) -> ModelProvenance | None:
        with self._lock:
            return self._models.get(model_id)

    def get_model_by_name_and_version(self, name: str, version: str) -> ModelProvenance | None:
        with self._lock:
            for model_id in self._versions.get(name, []):
                model = self._models.get(model_id)
                if model and model.version == version:
                    return model
            return None

    def get_latest_version(self, name: str) -> ModelProvenance | None:
        with self._lock:
            versions = self._versions.get(name, [])
            if not versions:
                return None
            latest_id = versions[-1]
            return self._models.get(latest_id)

    def list_versions(self, name: str) -> list[ModelProvenance]:
        with self._lock:
            return [
                self._models[mid] for mid in self._versions.get(name, []) if mid in self._models
            ]

    def rollback(self, name: str, target_version: str) -> tuple[bool, ModelProvenance | None]:
        """Rollback to a specific version."""
        with self._lock:
            target = self.get_model_by_name_and_version(name, target_version)
            if not target:
                return False, None

            # Create rollback record
            from .provenance import ProvenanceEventType

            latest = self.get_latest_version(name)
            target.add_event(
                ProvenanceEventType.MODIFIED,
                "system",
                f"Rollback to version {target_version}",
                metadata={"rollback_from": latest.version if latest else None},
            )

            self.trust_layer.create_record(
                RecordType.MODEL,
                target.__dict__,
                metadata={
                    "model_id": target.provenance_id,
                    "action": "rollback",
                    "target_version": target_version,
                },
            )

            return True, target

    def verify_model(self, model_id: str) -> tuple[bool, list[str]]:
        with self._lock:
            model = self._models.get(model_id)
            if not model:
                return False, ["Model not found"]

            trust_records = self.trust_layer.get_records_by_type(RecordType.MODEL)
            for record in trust_records:
                if record.payload.get("provenance_id") == model_id:
                    valid = self.trust_layer.verify_record(record)
                    if not valid:
                        return False, ["Trust record verification failed"]
            return True, []
