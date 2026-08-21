from __future__ import annotations

import json

import pytest

from cadgenesis.platform.config import ConfigStore, load_config


class TestConfigStore:
    def test_json_roundtrip(self, tmp_path):
        source = tmp_path / "platform.json"
        source.write_text(json.dumps({"serving": {"max_batch_size": 8}}), encoding="utf-8")
        store = ConfigStore([str(source)])
        assert store.get_nested("serving.max_batch_size") == 8

    def test_env_overlay(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CADGENESIS_MAX_BATCH", "16")
        source = tmp_path / "platform.json"
        source.write_text(json.dumps({"max_batch": 8}), encoding="utf-8")
        store = ConfigStore([str(source)], env_prefix="CADGENESIS_")
        assert store.get("max_batch") == 16

    def test_defaults_used(self, tmp_path):
        store = ConfigStore([], defaults={"mode": "prod"})
        assert store.get("mode") == "prod"

    def test_get_nested_missing(self, tmp_path):
        store = ConfigStore([])
        assert store.get("nope") is None
        assert store.get("nope", default="x") == "x"
        assert store.get_nested("a.b.c") is None

    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ConfigStore([str(tmp_path / "missing.json")])

    def test_reload(self, tmp_path):
        source = tmp_path / "platform.json"
        source.write_text(json.dumps({"a": 1}), encoding="utf-8")
        store = ConfigStore([str(source)])
        assert store.get("a") == 1
        source.write_text(json.dumps({"a": 2}), encoding="utf-8")
        assert store.reload() is True
        assert store.get("a") == 2

    def test_save(self, tmp_path):
        source = tmp_path / "platform.json"
        source.write_text(json.dumps({"x": [1, 2]}), encoding="utf-8")
        store = ConfigStore([str(source)])
        out = tmp_path / "out.json"
        store.save(out)
        assert json.loads(out.read_text(encoding="utf-8")) == {"x": [1, 2]}

    def test_unknown_format(self, tmp_path):
        source = tmp_path / "platform.ini"
        source.write_text("[x]", encoding="utf-8")
        with pytest.raises(ValueError):
            ConfigStore([str(source)])


class TestLoadConfig:
    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "missing.json"))

    def test_load_json(self, tmp_path):
        source = tmp_path / "c.json"
        source.write_text(json.dumps({"k": "v"}), encoding="utf-8")
        store = load_config(str(source))
        assert store.get("k") == "v"

    def test_load_single_path(self, tmp_path):
        source = tmp_path / "c.json"
        source.write_text(json.dumps({"k": "v"}), encoding="utf-8")
        assert load_config(source).get("k") == "v"

    def test_load_unknown_extension(self, tmp_path):
        source = tmp_path / "c.xyz"
        source.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(str(source))
