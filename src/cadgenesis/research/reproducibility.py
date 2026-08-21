"""
cadgenesis.research.reproducibility
===================================
Reproducibility toolkit for CADGenesis-LM research infrastructure.

- Deterministic training: seed management (Python/NumPy/torch/CUDA) and a
  training determinism context manager
- Environment capture: platform, package versions, env vars (redacted),
  git metadata and run command -> ``environment.json``
- Dependency tracking: pinned package versions + pip freeze capture
- Seed management: global seed registry with per-run serialization
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Self

logger = logging.getLogger("cadgenesis.research.reproducibility")

REDACT_TOKENS = ("KEY", "SECRET", "PASSWORD", "TOKEN", "AUTH")


def set_seed(seed: int, torch: Any | None = None) -> None:
    """Seed Python, random, numpy and (optionally) torch/CUDA deterministically."""
    random.seed(seed)
    import builtins

    if hasattr(builtins, "hash"):
        pass  # PYTHONHASHSEED is the authoritative python-level control
    try:
        import numpy as np  # type: ignore[import-not-found]

        np.random.seed(seed)
    except ImportError:
        pass
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class DeterministicTraining:
    """Context manager enforcing reproducible training settings."""

    def __init__(self, seed: int = 42, torch: Any | None = None) -> None:
        self.seed = seed
        self.torch = torch
        self._restore: list[tuple[str, Any]] = []

    def __enter__(self) -> Self:
        set_seed(self.seed, self.torch)
        if self.torch is not None:
            if hasattr(self.torch, "use_deterministic_algorithms"):
                self._restore.append(
                    (
                        "deterministic",
                        self.torch.are_deterministic_algorithms_enabled()
                        if hasattr(self.torch, "are_deterministic_algorithms_enabled")
                        else False,
                    )
                )
                self.torch.use_deterministic_algorithms(True)
            if hasattr(self.torch, "set_float32_matmul_precision"):
                self.torch.set_float32_matmul_precision("highest")
        return self

    def __exit__(self, *exc: object) -> None:
        if self.torch is not None:
            for name, value in self._restore:
                if name == "deterministic":
                    self.torch.use_deterministic_algorithms(bool(value))
        self._restore.clear()


@dataclass
class EnvironmentCapture:
    """A captured environment snapshot for reproducibility."""

    python_version: str = ""
    platform: str = ""
    packages: dict[str, str] = field(default_factory=dict)
    env_vars: dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    command: str = ""

    @classmethod
    def capture(cls, redact_env: bool = True) -> EnvironmentCapture:
        """Capture the current environment (package versions best-effort)."""
        packages: dict[str, str] = {}
        try:
            import importlib.metadata as metadata

            for dist in metadata.distributions():
                name = dist.metadata.get("Name") or "?"
                version = dist.version
                packages[name] = version
        except Exception:
            pass
        env_vars: dict[str, str] = {}
        for key, value in os.environ.items():
            if redact_env and any(t in key.upper() for t in REDACT_TOKENS):
                value = "***"
            env_vars[key] = value
        return cls(
            python_version=platform.python_version(),
            platform=platform.platform(),
            packages=packages,
            env_vars=env_vars,
            cwd=os.getcwd(),
            command=" ".join(sys.argv),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "cwd": self.cwd,
            "command": self.command,
            "package_count": len(self.packages),
            "packages": dict(sorted(self.packages.items())),
            "env_vars": self.env_vars,
        }

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path


def capture_pip_freeze(exclude: Iterable[str] = ()) -> str:
    """Capture ``pip freeze`` output minus excluded packages (best-effort)."""
    try:
        output = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=60
        )
        lines = [
            line
            for line in output.stdout.splitlines()
            if not any(line.startswith(ex) for ex in exclude)
        ]
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("pip freeze failed: %s", exc)
        return ""


class SeedRegistry:
    """Global seed manager producing reproducible derived seeds."""

    def __init__(self, base_seed: int = 42) -> None:
        self.base_seed = base_seed
        self._issued: dict[str, int] = {}

    def seed_for(self, key: str) -> int:
        """Deterministic per-key derived seed: base + stable hash offset."""
        if key not in self._issued:
            digest = sum(ord(c) for c in key)
            self._issued[key] = (self.base_seed + digest * 7919) % (2**31 - 1)
        return self._issued[key]

    def issued(self) -> dict[str, int]:
        return dict(self._issued)


__all__ = [
    "DeterministicTraining",
    "EnvironmentCapture",
    "SeedRegistry",
    "capture_pip_freeze",
    "set_seed",
]
