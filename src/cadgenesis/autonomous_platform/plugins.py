"""
Enterprise Plugin Framework - CAD plugins, AI plugins, simulation plugins, manufacturing
plugins, enterprise integrations.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any


class PluginType(str, Enum):
    CAD = "cad"
    AI = "ai"
    SIMULATION = "simulation"
    MANUFACTURING = "manufacturing"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class PluginStatus(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginManifest:
    """Plugin manifest with metadata."""

    plugin_id: str
    name: str
    version: str
    plugin_type: PluginType
    description: str
    author: str
    license: str
    entry_point: str  # module:function
    dependencies: dict[str, str] = field(default_factory=dict)  # name -> version
    provides: list[str] = field(default_factory=list)  # capabilities
    requires: list[str] = field(default_factory=list)  # required capabilities
    config_schema: dict[str, Any] = field(default_factory=dict)
    homepage: str = ""
    repository: str = ""
    checksum: str = ""
    signature: str = ""
    signer_public_key: str = ""
    status: PluginStatus = PluginStatus.UNLOADED
    loaded_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginInstance:
    """Loaded plugin instance."""

    manifest: PluginManifest
    module: Any
    instance: Any
    config: dict[str, Any] = field(default_factory=dict)


class PluginRegistry:
    """Registry for managing plugins."""

    def __init__(self):
        self._manifests: dict[str, PluginManifest] = {}
        self._instances: dict[str, PluginInstance] = {}
        self._capability_index: dict[str, list[str]] = {}  # capability -> plugin_ids
        self._lock = RLock()

    def register_manifest(self, manifest: PluginManifest) -> bool:
        with self._lock:
            if manifest.plugin_id in self._manifests:
                return False
            self._manifests[manifest.plugin_id] = manifest
            for cap in manifest.provides:
                if cap not in self._capability_index:
                    self._capability_index[cap] = []
                self._capability_index[cap].append(manifest.plugin_id)
            return True

    def get_manifest(self, plugin_id: str) -> PluginManifest | None:
        with self._lock:
            return self._manifests.get(plugin_id)

    def find_by_capability(self, capability: str) -> list[PluginManifest]:
        with self._lock:
            plugin_ids = self._capability_index.get(capability, [])
            return [self._manifests[pid] for pid in plugin_ids if pid in self._manifests]

    def list_plugins(
        self, plugin_type: PluginType | None = None, status: PluginStatus | None = None
    ) -> list[PluginManifest]:
        with self._lock:
            plugins = list(self._manifests.values())
            if plugin_type:
                plugins = [p for p in plugins if p.plugin_type == plugin_type]
            if status:
                plugins = [p for p in plugins if p.status == status]
            return plugins


class EnterprisePluginFramework:
    """Enterprise plugin framework for CADGenesis."""

    def __init__(self, plugin_dirs: list[str] | None = None):
        self.registry = PluginRegistry()
        self.plugin_dirs = [
            Path(d) for d in (plugin_dirs or ["./plugins", "/etc/cadgenesis/plugins"])
        ]
        self._instances: dict[str, PluginInstance] = {}
        self._lock = RLock()

    def discover_plugins(self) -> int:
        """Discover plugins from plugin directories."""
        count = 0
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue
            for plugin_file in plugin_dir.glob("*.py"):
                if plugin_file.name.startswith("_"):
                    continue
                try:
                    manifest = self._load_manifest(plugin_file)
                    if manifest and self.registry.register_manifest(manifest):
                        count += 1
                except Exception:
                    pass
        return count

    def _load_manifest(self, plugin_file: Path) -> PluginManifest | None:
        """Load plugin manifest from file."""
        spec = importlib.util.spec_from_file_location("plugin", plugin_file)
        if not spec or not spec.loader:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Look for PLUGIN_MANIFEST attribute
        if hasattr(module, "PLUGIN_MANIFEST"):
            return module.PLUGIN_MANIFEST
        return None

    def load_plugin(self, plugin_id: str, config: dict[str, Any] | None = None) -> bool:
        """Load and initialize a plugin."""
        with self._lock:
            manifest = self.registry.get_manifest(plugin_id)
            if not manifest:
                return False

            if manifest.status == PluginStatus.LOADED:
                return True

            manifest.status = PluginStatus.LOADING

            try:
                # Load module
                module = importlib.import_module(manifest.entry_point.split(":")[0])

                # Get factory function
                factory_name = (
                    manifest.entry_point.split(":")[1]
                    if ":" in manifest.entry_point
                    else "create_plugin"
                )
                factory = getattr(module, factory_name)

                # Create instance
                instance = factory(config or {})

                # Store instance
                plugin_instance = PluginInstance(
                    manifest=manifest,
                    module=module,
                    instance=instance,
                    config=config or {},
                )
                self._instances[plugin_id] = plugin_instance

                manifest.status = PluginStatus.LOADED
                manifest.loaded_at = time.time()

                return True
            except Exception as e:
                manifest.status = PluginStatus.ERROR
                manifest.metadata["error"] = str(e)
                return False

    def unload_plugin(self, plugin_id: str) -> bool:
        with self._lock:
            if plugin_id not in self._instances:
                return False

            instance = self._instances[plugin_id]
            # Call cleanup if available
            if hasattr(instance.instance, "cleanup"):
                with contextlib.suppress(Exception):
                    instance.instance.cleanup()

            del self._instances[plugin_id]
            manifest = self.registry.get_manifest(plugin_id)
            if manifest:
                manifest.status = PluginStatus.UNLOADED
                manifest.loaded_at = None
            return True

    def get_plugin(self, plugin_id: str) -> Any | None:
        with self._lock:
            instance = self._instances.get(plugin_id)
            return instance.instance if instance else None

    def get_plugins_by_capability(self, capability: str) -> list[Any]:
        """Get all loaded plugins providing a capability."""
        with self._lock:
            manifests = self.registry.find_by_capability(capability)
            return [
                self._instances[manifest.plugin_id].instance
                for manifest in manifests
                if manifest.plugin_id in self._instances
                and self._instances[manifest.plugin_id].manifest.status == PluginStatus.LOADED
            ]

    def list_loaded_plugins(self) -> list[PluginManifest]:
        with self._lock:
            return [inst.manifest for inst in self._instances.values()]
