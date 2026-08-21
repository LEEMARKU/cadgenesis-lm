from __future__ import annotations

import json

from cadgenesis.research.reproducibility import (
    DeterministicTraining,
    EnvironmentCapture,
    SeedRegistry,
    capture_pip_freeze,
    set_seed,
)


class TestSetSeed:
    def test_runs(self):
        set_seed(42)
        set_seed(7)
        set_seed(1, torch=None)

    def test_torch_seeded(self):
        try:
            import torch
        except ImportError:
            return
        set_seed(5, torch)
        first = torch.randint(0, 10**6, (1,)).item()
        set_seed(5, torch)
        second = torch.randint(0, 10**6, (1,)).item()
        assert first == second


class TestDeterministicTraining:
    def test_context(self):
        try:
            import torch
        except ImportError:
            return
        with DeterministicTraining(seed=42, torch=torch):
            torch.randn(3)
        with DeterministicTraining(seed=42, torch=torch):
            torch.randn(3)

    def test_context_without_torch(self):
        with DeterministicTraining(seed=42):
            pass


class TestEnvironmentCapture:
    def test_capture_shape(self):
        env = EnvironmentCapture.capture()
        data = env.to_dict()
        assert data["python_version"]
        assert data["platform"]
        assert "packages" in data

    def test_redaction(self, monkeypatch):
        monkeypatch.setenv("MY_API_KEY", "super-secret")
        env = EnvironmentCapture.capture()
        assert env.env_vars.get("MY_API_KEY") == "***"

    def test_save(self, tmp_path):
        path = EnvironmentCapture.capture().save(str(tmp_path / "env.json"))
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        assert "python_version" in payload


class TestCapturePipFreeze:
    def test_runs(self):
        output = capture_pip_freeze(exclude=("pytest",))
        assert isinstance(output, str)


class TestSeedRegistry:
    def test_deterministic_per_key(self):
        registry = SeedRegistry(base_seed=42)
        assert registry.seed_for("a") == registry.seed_for("a")
        assert registry.seed_for("a") != registry.seed_for("b")
        assert "a" in registry.issued()
