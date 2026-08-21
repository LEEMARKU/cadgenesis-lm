"""
cadgenesis.platform.plugins
===========================
Runtime plugin system for the CADGenesis-LM platform.

- Discovery: scan plugin directories for ``plugin.py``/``plugin.json``
- Loading: ``importlib`` based, isolated namespace per plugin
- Dependency validation: plugin A -> plugin B (versions + ordering)
- Version compatibility: ``requires_cadgenesis`` and ``plugin_version`` checks

Plugin classes subclass :class:`PlatformPlugin` and implement the ``activate``
hook.  Compatibility mirrors the existing transformer/agent plugin patterns.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cadgenesis import __version__ as CADGENESIS_VERSION

logger = logging.getLogger("cadgenesis.platform.plugins")


class PluginError(Exception):
    """Plugin discovery/load/activation failure."""


@dataclass(frozen=True)
class PluginManifest:
    """Metadata for a plugin (from ``plugin.json`` or module attributes)."""

    name: str
    version: str = "1.0.0"
    description: str = ""
    entry_point: str = "plugin.py"
    dependencies: tuple[str, ...] = ()
    requires_cadgenesis: str | None = None
    author: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], entry_point: str = "plugin.py") -> PluginManifest:
        return cls(
            name=str(data["name"]),
            version=str(data.get("version", "1.0.0")),
            description=str(data.get("description", "")),
            entry_point=str(data.get("entry_point", entry_point)),
            dependencies=tuple(data.get("dependencies", [])),
            requires_cadgenesis=data.get("requires_cadgenesis"),
            author=str(data.get("author", "")),
        )


class PlatformPlugin:
    """Base class for platform plugins."""

    manifest: PluginManifest | None = None

    def activate(self, context: dict[str, Any]) -> None:
        """Hook invoked after the plugin is loaded; receives shared context."""

    def deactivate(self) -> None:
        """Hook invoked when the plugin is unloaded."""


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for segment in version.replace("-", ".").split("."):
        try:
            parts.append(int(segment))
        except ValueError:
            break
    return tuple(parts)


def version_compatible(plugin_version: str, required: str) -> bool:
    """True when ``plugin_version`` >= ``required`` (simple semver compare)."""
    return _version_tuple(plugin_version) >= _version_tuple(required)


class PluginManager:
    """Discovers, validates, loads and activates plugins from directories."""

    def __init__(
        self, directories: Iterable[str | Path] = (), min_cadgenesis: str | None = None
    ) -> None:
        self.directories = [Path(d) for d in directories]
        self.min_cadgenesis = min_cadgenesis
        self._manifests: dict[str, PluginManifest] = {}
        self._modules: dict[str, Any] = {}
        self._plugins: dict[str, PlatformPlugin] = {}
        self._context: dict[str, Any] = {}

    # ------------------------------------------------------------ discovery

    def discover(self) -> list[PluginManifest]:
        """Scan directories for ``plugin.json`` or ``plugin.py`` entries."""
        found: dict[str, PluginManifest] = {}
        for directory in self.directories:
            if not directory.exists():
                continue

            def _register(name: str, plugin_dir: Path) -> None:
                manifest_file = plugin_dir / "plugin.json"
                if manifest_file.exists():
                    try:
                        data = json.loads(manifest_file.read_text(encoding="utf-8"))
                        found[name] = PluginManifest.from_dict(data, entry_point="plugin.py")
                        return
                    except (ValueError, KeyError) as exc:
                        raise PluginError(f"invalid manifest {manifest_file}: {exc}") from exc
                found[name] = PluginManifest(name=name, entry_point="plugin.py")

            for manifest_file in sorted(directory.glob("plugin.json")):
                try:
                    data = json.loads(manifest_file.read_text(encoding="utf-8"))
                    manifest = PluginManifest.from_dict(data, entry_point="plugin.py")
                except (ValueError, KeyError) as exc:
                    raise PluginError(f"invalid manifest {manifest_file}: {exc}") from exc
                found[manifest.name] = manifest
            for entry in sorted(directory.iterdir()):
                if entry.is_dir():
                    if (entry / "plugin.py").exists():
                        _register(entry.name, entry)
                elif entry.name == "plugin.py":
                    _register(entry.parent.name, entry.parent)
        self._manifests = found
        return list(found.values())

    # ------------------------------------------------------------ validation

    def validate_dependencies(self, manifests: Iterable[PluginManifest] | None = None) -> None:
        """Every declared dependency must be present with a compatible version."""
        available = {m.name: m for m in (manifests or self._manifests.values())}
        for manifest in available.values():
            for dependency in manifest.dependencies:
                dep_name, _, min_version = dependency.partition(">=")
                if dep_name not in available:
                    raise PluginError(
                        f"plugin {manifest.name!r} requires missing dependency {dep_name!r}"
                    )
                if min_version and not version_compatible(available[dep_name].version, min_version):
                    raise PluginError(
                        f"plugin {manifest.name!r} requires {dep_name}>={min_version}, "
                        f"found {available[dep_name].version}"
                    )

    def check_cadgenesis_version(self, manifest: PluginManifest) -> None:
        required = manifest.requires_cadgenesis or self.min_cadgenesis
        if required and not version_compatible(CADGENESIS_VERSION, required):
            raise PluginError(
                f"plugin {manifest.name!r} requires cadgenesis>={required},"
                f" running {CADGENESIS_VERSION}"
            )

    # ---------------------------------------------------------------- load

    def load(self, name: str, directory: Path | None = None) -> PlatformPlugin:
        """Load and activate one plugin by name."""
        if name in self._plugins:
            return self._plugins[name]
        if not self._manifests:
            self.discover()
        manifest = self._manifests.get(name)
        if manifest is None:
            raise PluginError(f"plugin {name!r} not found")
        self.check_cadgenesis_version(manifest)
        for dependency in manifest.dependencies:
            dep_name = dependency.partition(">=")[0]
            if dep_name not in self._plugins:
                self.load(dep_name)
        source_dir = directory or self._resolve_directory(manifest)
        module = self._import_plugin(name, source_dir, manifest.entry_point)
        plugin_instance = self._instantiate(module, manifest)
        self._plugins[name] = plugin_instance
        try:
            plugin_instance.activate(self._context)
        except Exception as exc:
            del self._plugins[name]
            raise PluginError(f"activation failed for plugin {name!r}: {exc}") from exc
        logger.info("plugin loaded: %s v%s", name, manifest.version)
        return plugin_instance

    def _resolve_directory(self, manifest: PluginManifest) -> Path:
        for directory in self.directories:
            candidate = directory / manifest.name
            if candidate.exists() or (directory / manifest.entry_point).exists():
                return candidate if candidate.exists() else directory
        raise PluginError(f"cannot locate sources for plugin {manifest.name!r}")

    @staticmethod
    def _import_plugin(name: str, directory: Path, entry_point: str) -> Any:
        module_path = directory / entry_point
        if not module_path.exists():
            module_path = directory / "plugin.py"
        if not module_path.exists():
            raise PluginError(f"plugin {name!r} has no {entry_point} entry point")
        module_name = f"_cadgenesis_plugin_{name.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise PluginError(f"cannot build import spec for plugin {name!r}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _instantiate(self, module: Any, manifest: PluginManifest) -> PlatformPlugin:
        plugin_class = getattr(module, "Plugin", None)
        if plugin_class is None:
            for value in vars(module).values():
                if (
                    isinstance(value, type)
                    and issubclass(value, PlatformPlugin)
                    and value is not PlatformPlugin
                ):
                    plugin_class = value
                    break
        if plugin_class is None:
            raise PluginError(f"plugin {manifest.name!r} defines no PlatformPlugin subclass")
        instance = plugin_class()
        instance.manifest = manifest
        return instance

    # ---------------------------------------------------------------- manage

    def load_all(self) -> list[PlatformPlugin]:
        manifests = self.discover()
        self.validate_dependencies(manifests)
        return [self.load(m.name) for m in manifests]

    def unload(self, name: str) -> None:
        plugin = self._plugins.pop(name, None)
        if plugin is not None:
            plugin.deactivate()
        self._modules.pop(name, None)
        logger.info("plugin unloaded: %s", name)

    def get(self, name: str) -> PlatformPlugin | None:
        return self._plugins.get(name)

    def active(self) -> list[str]:
        return sorted(self._plugins)

    def share(self, key: str, value: Any) -> None:
        """Inject shared context visible to all plugins."""
        self._context[key] = value

    def __contains__(self, name: str) -> bool:
        return name in self._plugins


__all__ = ["PlatformPlugin", "PluginError", "PluginManager", "PluginManifest", "version_compatible"]
