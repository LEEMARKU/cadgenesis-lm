from __future__ import annotations

import pytest

from cadgenesis.platform.registry import ModelRegistry, parse_version


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path / "registry")


class TestRegistryVersioning:
    def test_register_default_version(self, registry):
        record = registry.register("cad", "some/path.pt")
        assert record.name == "cad"
        assert record.version == "v1"

    def test_auto_increment(self, registry):
        registry.register("cad", "p.pt")
        record = registry.register("cad", "p.pt")
        assert record.version == "v2"

    def test_duplicate_version_rejected(self, registry):
        registry.register("cad", "p.pt")
        with pytest.raises(ValueError):
            registry.register("cad", "p.pt", version="v1")

    def test_get_by_version_and_alias(self, registry):
        registry.register("cad", "p.pt")
        registry.register("cad", "p.pt")
        assert registry.get("cad", version="v1").version == "v1"
        assert registry.get("cad").version == "v2"
        assert registry.get("cad", alias="latest").version == "v2"

    def test_get_missing(self, registry):
        assert registry.get("nope") is None


class TestRegistryOps:
    def test_promote_and_deployment_history(self, registry):
        registry.register("cad", "p.pt")
        registry.register("cad", "p.pt")
        registry.promote("cad", "v2", environment="production", actor="alice")
        assert registry.get("cad", alias="production").version == "v2"
        history = registry.deployment_history("cad")
        assert history[-1].actor == "alice"
        assert history[-1].status == "promoted"

    def test_rollback(self, registry):
        registry.register("cad", "p.pt")
        registry.register("cad", "p.pt")
        registry.promote("cad", "v2")
        registry.promote("cad", "v1")
        assert registry.get("cad", alias="production").version == "v1"
        rolled_back = registry.rollback("cad")
        assert rolled_back.version == "v2"
        assert registry.get("cad", alias="production").version == "v2"

    def test_rollback_without_history(self, registry):
        registry.register("cad", "p.pt")
        registry.promote("cad", "v1")
        assert registry.rollback("cad") is None

    def test_set_alias(self, registry):
        registry.register("cad", "p.pt")
        registry.register("cad", "p.pt")
        registry.set_alias("cad", "canary", "v1")
        assert registry.get("cad", alias="canary").version == "v1"
        with pytest.raises(ValueError):
            registry.set_alias("cad", "x", "v99")


class TestRegistryPersistence:
    def test_reload(self, tmp_path):
        path = tmp_path / "registry"
        registry = ModelRegistry(path)
        registry.register("cad", "p.pt", metadata={"metric": 1.0})
        registry.promote("cad", "v1", environment="staging")
        reloaded = ModelRegistry(path)
        assert reloaded.get("cad").version == "v1"
        assert reloaded.get("cad", alias="staging").version == "v1"

    def test_list(self, registry):
        registry.register("a", "p.pt")
        registry.register("b", "p.pt")
        assert registry.list_models() == ["a", "b"]
        assert len(registry.list_versions("a")) == 1


class TestParseVersion:
    def test_parse(self):
        assert parse_version("v12") == (12,)
        assert parse_version("v1.2") == (1, 2)
