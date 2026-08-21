"""
cadgenesis.serving.lifecycle
============================
Model lifecycle management for the CADGenesis serving stack.

- Load models from checkpoints/registry with warm-up and health probes
- Multi-model registry mapping (name -> engine), hot (re)load and unload
- Idempotent load, guard against double-loads, and health state tracking
- Integrates with ``cadgenesis.platform.registry.ModelRegistry`` for the
  "which version to serve" resolution (``production`` alias)
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("cadgenesis.serving.lifecycle")

LoadFn = Callable[[str], Any]
WarmupFn = Callable[[Any], None]


@dataclass
class ServedModel:
    """One loaded model instance served by the platform."""

    name: str
    engine: Any
    path: str
    loaded_at: float
    version: str | None = None
    healthy: bool = True
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "version": self.version,
            "loaded_at": self.loaded_at,
            "healthy": self.healthy,
            "last_error": self.last_error,
        }


class ModelLifecycle:
    """Thread-safe registry of loaded models with reload/unload support."""

    def __init__(self, load_fn: LoadFn, warmup: WarmupFn | None = None) -> None:
        self.load_fn = load_fn
        self.warmup = warmup
        self._models: dict[str, ServedModel] = {}
        self._lock = threading.RLock()

    def load(
        self,
        name: str,
        path: str,
        version: str | None = None,
        force: bool = False,
    ) -> ServedModel:
        """Load (or reload) ``name`` from ``path``. Idempotent unless ``force``."""
        with self._lock:
            existing = self._models.get(name)
            if existing is not None and not force and existing.path == path:
                return existing
            started = time.time()
            try:
                engine = self.load_fn(path)
                if self.warmup is not None:
                    self.warmup(engine)
            except Exception as exc:
                logger.exception("failed to load model %s from %s", name, path)
                self._models[name] = ServedModel(
                    name=name,
                    engine=None,
                    path=path,
                    loaded_at=started,
                    version=version,
                    healthy=False,
                    last_error=str(exc),
                )
                raise
            model = ServedModel(
                name=name, engine=engine, path=path, loaded_at=time.time(), version=version
            )
            self._models[name] = model
            logger.info("model %s loaded from %s (%.1fs)", name, path, time.time() - started)
            return model

    def unload(self, name: str) -> bool:
        with self._lock:
            return self._models.pop(name, None) is not None

    def get(self, name: str = "default") -> ServedModel | None:
        with self._lock:
            return self._models.get(name)

    def engine(self, name: str = "default") -> Any:
        model = self.get(name)
        if model is None or model.engine is None:
            raise KeyError(f"model {name!r} not loaded")
        return model.engine

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._models)

    def status(self) -> list[dict[str, Any]]:
        with self._lock:
            return [m.to_dict() for m in self._models.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._models


def resolve_registry_path(
    registry_directory: str | None,
    name: str,
    environment: str = "production",
) -> str | None:
    """Resolve the checkpoint path for a registered model version (or None)."""
    if not registry_directory:
        return None
    from cadgenesis.platform.registry import ModelRegistry

    registry = ModelRegistry(registry_directory)
    record = registry.get(name, alias=environment)
    if record is None:
        return None
    if not Path(record.path).exists():
        logger.warning("registry path %s for %s missing", record.path, name)
    return record.path


__all__ = ["ModelLifecycle", "ServedModel", "resolve_registry_path"]
