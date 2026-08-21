"""
cadgenesis.platform.config
==========================
Multi-format configuration system for the CADGenesis-LM production platform.

Supports JSON, YAML (optional ``pyyaml``), TOML (3.11+ stdlib ``tomllib``,
3.10 via optional ``tomli``) and environment-variable overlay, with dynamic
reloading: ``ConfigStore`` watches source files by content digest and
re-reads on request.  The existing ``CADConfig`` remains the single source
of truth for model/training settings; this module manages *platform*
settings (serving, auth, registry, monitoring) plus generic key/value
layers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger("cadgenesis.platform.config")

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore[import-not-found]
except ImportError:
    yaml = None  # type: ignore[assignment]

try:  # pragma: no cover - stdlib in 3.11+
    import tomllib as _tomllib
except ImportError:  # pragma: no cover
    try:
        import tomli as _tomllib  # type: ignore[no-redef, import-not-found]
    except ImportError:
        _tomllib = None  # type: ignore[assignment]

SUPPORTED_FORMATS = ("json", "yaml", "toml")


def _load_bytes(path: Path, raw: bytes) -> dict[str, Any]:
    """Parse a JSON/YAML/TOML document from raw bytes (format by extension)."""
    suffix = path.suffix.lower().lstrip(".")
    if suffix in ("yaml", "yml"):
        if yaml is None:
            raise RuntimeError("YAML support requires 'pyyaml' (pip install pyyaml)")
        data = yaml.safe_load(raw) or {}
    elif suffix == "toml":
        if _tomllib is None:
            raise RuntimeError("TOML support requires Python 3.11+ or 'tomli'")
        data = _tomllib.loads(raw.decode("utf-8"))
    elif suffix == "json":
        data = json.loads(raw.decode("utf-8"))
    else:
        raise ValueError(f"unsupported config format {suffix!r}; expected {SUPPORTED_FORMATS}")
    if not isinstance(data, dict):
        raise ValueError(f"config root must be an object, got {type(data).__name__}")
    return data


def _flatten(
    data: Mapping[str, Any],
    prefix: str = "",
    separator: str = "__",
) -> dict[str, Any]:
    """Flatten nested dicts: ``{"server": {"port": 1}}`` -> ``{"server__port": 1}``."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full = f"{prefix}{separator}{key}" if prefix else key
        if isinstance(value, Mapping) and not isinstance(value, (str, bytes)):
            flat.update(_flatten(value, prefix=full, separator=separator))
        else:
            flat[full] = value
    return flat


def env_prefix_overlay(prefix: str = "CADGENESIS_") -> dict[str, str]:
    """Env-var overlay: ``CADGENESIS_SERVER__PORT=8080`` -> ``server__port``."""
    overlay: dict[str, str] = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            name = key[len(prefix) :].lower()
            overlay[name] = value
    return overlay


def coerce_value(raw: str) -> Any:
    """Best-effort scalar coercion for env-var overlays."""
    lowered = raw.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("none", "null"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


class ConfigStore:
    """Layered, reloadable configuration with env overlay and change hooks."""

    def __init__(
        self,
        sources: list[str | os.PathLike[str]] | str | os.PathLike[str] | None = None,
        env_prefix: str = "CADGENESIS_",
        defaults: Mapping[str, Any] | None = None,
    ) -> None:
        if sources is None:
            source_list: list[str] = []
        elif isinstance(sources, (str, os.PathLike)):
            source_list = [str(sources)]
        else:
            source_list = [str(s) for s in sources]
        self.sources = source_list
        self.env_prefix = env_prefix
        self._defaults = dict(defaults or {})
        self._data: dict[str, Any] = {}
        self._digests: dict[str, str] = {}
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self._lock = threading.RLock()
        self.reload()

    # ------------------------------------------------------------------ io

    def reload(self) -> bool:
        """Re-read all sources; returns True when anything changed."""
        with self._lock:
            changed = False
            combined: dict[str, Any] = {}
            for source in self.sources:
                path = Path(source)
                if not path.exists():
                    raise FileNotFoundError(f"config source not found: {source}")
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if self._digests.get(source) != digest:
                    changed = True
                    self._digests[source] = digest
                combined.update(_load_bytes(path, raw))
            env = env_prefix_overlay(self.env_prefix)
            env = {k: coerce_value(v) for k, v in env.items()}
            if env:
                changed = True
            merged = dict(self._defaults)
            merged.update(combined)
            merged.update(env)
            self._data = merged
            if changed:
                for listener in list(self._listeners):
                    try:
                        listener(dict(self._data))
                    except Exception:
                        logger.exception("config reload listener failed")
            return changed

    def watch(self) -> bool:
        """Alias for :meth:`reload` used by periodic watchers (dynamic reload)."""
        return self.reload()

    # -------------------------------------------------------------- access

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def get_nested(self, dotted_key: str, default: Any = None) -> Any:
        """Access via dotted path (``server.port``) or flattened key (``server__port``)."""
        value = self.get(dotted_key, ...)
        if value is not ...:
            return value
        current: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        return current

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def as_dict(self, flatten: bool = False) -> dict[str, Any]:
        with self._lock:
            return _flatten(self._data) if flatten else dict(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def on_change(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback invoked whenever a reload changes data."""
        with self._lock:
            self._listeners.append(callback)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Persist current data (JSON or YAML, by extension) to ``path``."""
        target = Path(path)
        suffix = target.suffix.lower().lstrip(".")
        if suffix in ("yaml", "yml"):
            if yaml is None:
                raise RuntimeError("YAML support requires 'pyyaml'")
            with target.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(self._data, handle, sort_keys=False)
        elif suffix == "toml":
            raise ValueError("TOML writing is not supported; use JSON or YAML")
        else:
            with target.open("w", encoding="utf-8") as handle:
                json.dump(self._data, handle, indent=2)
                handle.write("\n")


def load_config(
    sources: list[str | os.PathLike[str]] | str | os.PathLike[str] | None = None,
    env_prefix: str = "CADGENESIS_",
    reload: bool = False,
    defaults: Mapping[str, Any] | None = None,
) -> ConfigStore:
    """Convenience factory: build a :class:`ConfigStore` from files + env."""
    store = ConfigStore(sources=sources, env_prefix=env_prefix, defaults=defaults)
    if reload:
        logger.info("dynamic reload enabled for %d source(s)", len(store.sources))
    return store


__all__ = ["SUPPORTED_FORMATS", "ConfigStore", "coerce_value", "env_prefix_overlay", "load_config"]
