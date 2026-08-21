from __future__ import annotations

import pytest

from cadgenesis.platform.plugins import (
    PlatformPlugin,
    PluginError,
    PluginManager,
    PluginManifest,
    version_compatible,
)


class TestPluginManifest:
    def test_from_dict(self):
        manifest = PluginManifest.from_dict(
            {
                "name": "p",
                "version": "1.0.0",
                "entry_point": "plugin.py",
                "dependencies": ["q>=1.0.0"],
            }
        )
        assert manifest.name == "p"
        assert manifest.dependencies == ("q>=1.0.0",)


class TestVersionCompatible:
    def test_compare(self):
        assert version_compatible("6.1.0", "6.0.0")
        assert not version_compatible("5.9.0", "6.0.0")
        assert version_compatible("6.0.0", "6.0.0")


PLUGIN_SOURCE = (
    "from cadgenesis.platform.plugins import PlatformPlugin\n"
    "class Plugin(PlatformPlugin):\n"
    "    def activate(self, context):\n"
    "        context['calls'] = context.get('calls', 0) + 1\n"
    "    def deactivate(self):\n"
    "        pass\n"
)


class TestPluginManager:
    def test_discover_and_load(self, tmp_path):
        plugin_dir = tmp_path / "hello_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.py").write_text(PLUGIN_SOURCE, encoding="utf-8")
        manager = PluginManager([str(tmp_path)])
        manifests = manager.discover()
        assert any(m.name == "hello_plugin" for m in manifests)
        plugin = manager.load("hello_plugin")
        assert isinstance(plugin, PlatformPlugin)
        assert "hello_plugin" in manager.active()
        assert manager.get("hello_plugin") is plugin
        assert "hello_plugin" in manager

    def test_shared_context(self, tmp_path):
        plugin_dir = tmp_path / "ctx_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.py").write_text(PLUGIN_SOURCE, encoding="utf-8")
        manager = PluginManager([str(tmp_path)])
        manager.discover()
        manager.load("ctx_plugin")
        manager.unload("ctx_plugin")
        assert "ctx_plugin" not in manager.active()

    def test_manifest_json(self, tmp_path):
        plugin_dir = tmp_path / "json_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.py").write_text(PLUGIN_SOURCE, encoding="utf-8")
        (plugin_dir / "plugin.json").write_text(
            '{"name": "json_plugin", "version": "1.2.3", "dependencies": []}', encoding="utf-8"
        )
        manager = PluginManager([str(tmp_path)])
        manifests = manager.discover()
        manifest = next(m for m in manifests if m.name == "json_plugin")
        assert manifest.version == "1.2.3"

    def test_dependency_validation(self):
        a = PluginManifest(name="a", dependencies=("missing>=1.0.0",))
        manager = PluginManager()
        with pytest.raises(PluginError):
            manager.validate_dependencies([a])

    def test_load_missing(self, tmp_path):
        manager = PluginManager([str(tmp_path)])
        with pytest.raises(PluginError):
            manager.load("not_there")

    def test_plugin_without_subclass(self, tmp_path):
        plugin_dir = tmp_path / "bad_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.py").write_text("x = 1\n", encoding="utf-8")
        manager = PluginManager([str(tmp_path)])
        manager.discover()
        with pytest.raises(PluginError):
            manager.load("bad_plugin")
