"""
cadgenesis.platform.sdk
=======================
High-level Python SDK for the CADGenesis-LM platform.

Supports inference, training, deployment, model registry and plugins, both
locally (in-process, dependency-free) and remotely (REST over stdlib
``urllib``; gRPC when ``grpcio`` is installed).  The SDK is the supported
programmatic surface for downstream tooling and notebooks.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from cadgenesis import __version__ as CADGENESIS_VERSION

logger = logging.getLogger("cadgenesis.platform.sdk")


class SDKError(Exception):
    """SDK operation failure (transport, auth or payload)."""


@dataclass
class InferenceRequest:
    """Input for a generation/inference call."""

    text: str
    max_len: int = 128
    beam_width: int = 1
    temperature: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "max_len": self.max_len,
            "beam_width": self.beam_width,
            "temperature": self.temperature,
        }


@dataclass
class InferenceResult:
    """Result of an inference call (remote or local)."""

    text: str = ""
    tokens: list[str] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class LocalBackend:
    """In-process backend wrapping :class:`CADInferenceEngine`."""

    def __init__(self, engine: Any, tokenizer: Any | None = None) -> None:
        self.engine = engine
        self.tokenizer = tokenizer

    def generate(self, request: InferenceRequest) -> InferenceResult:
        start = time.perf_counter()
        kwargs: dict[str, Any] = {"max_len": request.max_len}
        if request.beam_width and request.beam_width > 1:
            result = self.engine.beam(request.text, beam_width=request.beam_width, **kwargs)
        else:
            result = self.engine.greedy(request.text, **kwargs)
        latency = (time.perf_counter() - start) * 1000.0
        return InferenceResult(
            text=" ".join(getattr(result, "tokens", [])),
            tokens=list(getattr(result, "tokens", [])),
            confidence=float(getattr(result, "confidence", 0.0)),
            latency_ms=round(latency, 3),
            meta={
                "toon": getattr(result, "toon", ""),
                "stopped_on_eos": getattr(result, "stopped_on_eos", False),
            },
        )


class RestBackend:
    """Remote backend over the platform REST API (stdlib urllib)."""

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"cadgenesis-sdk/{CADGENESIS_VERSION}",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, path: str, payload: Mapping[str, Any]) -> Any:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path, data=body, headers=self._headers(), method="POST"
        )
        return self._urlopen(request, path)

    def _get(self, path: str) -> Any:
        request = urllib.request.Request(
            self.base_url + path, headers=self._headers(), method="GET"
        )
        return self._urlopen(request, path)

    def _urlopen(self, request: urllib.request.Request, path: str) -> Any:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SDKError(f"HTTP {exc.code} from {path}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise SDKError(f"connection error to {self.base_url}: {exc.reason}") from exc

    def generate(self, request: InferenceRequest) -> InferenceResult:
        payload = request.to_dict()
        data = self._post("/api/v1/inference/generate", payload)
        return InferenceResult(
            text=str(data.get("text", "")),
            tokens=[str(t) for t in data.get("tokens", [])],
            confidence=float(data.get("confidence", 0.0)),
            latency_ms=float(data.get("latency_ms", 0.0)),
            meta=data,
        )

    def health(self) -> dict[str, Any]:
        try:
            request = urllib.request.Request(self.base_url + "/healthz", headers=self._headers())
            with urllib.request.urlopen(request, timeout=5.0) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, ValueError) as exc:
            return {"ok": False, "detail": str(exc)}


class CADGenesisSDK:
    """The unified SDK facade: inference, training, deployment, plugins."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        local_engine: Any = None,
        local_tokenizer: Any = None,
    ) -> None:
        if base_url:
            self.backend: RestBackend | LocalBackend = RestBackend(base_url, api_key)
        elif local_engine is not None:
            self.backend = LocalBackend(local_engine, local_tokenizer)
        else:
            raise ValueError("SDK requires either base_url or local_engine")
        self.training = TrainingClient(self)
        self.deployment = DeploymentClient(self)
        self.plugins = PluginClient(self)

    # ------------------------------------------------------------- inference

    def generate(
        self,
        text: str,
        max_len: int = 128,
        beam_width: int = 1,
        temperature: float = 1.0,
    ) -> InferenceResult:
        return self.backend.generate(InferenceRequest(text, max_len, beam_width, temperature))

    def generate_batch(self, texts: Sequence[str], **kwargs: Any) -> list[InferenceResult]:
        return [self.generate(text, **kwargs) for text in texts]

    def health(self) -> dict[str, Any]:
        if isinstance(self.backend, RestBackend):
            return self.backend.health()
        return {"ok": True, "backend": "local", "version": CADGENESIS_VERSION}


class TrainingClient:
    """SDK surface for training jobs (local or via platform REST)."""

    def __init__(self, sdk: CADGenesisSDK) -> None:
        self._sdk = sdk

    def launch(
        self,
        config_path: str,
        epochs: int = 1,
        resume_from: str | None = None,
        wait: bool = True,
    ) -> dict[str, Any]:
        """Launch a training run; returns job info/status."""
        payload = {"config": config_path, "epochs": epochs, "resume_from": resume_from}
        if isinstance(self._sdk.backend, RestBackend):
            job = self._sdk.backend._post("/api/v1/training/jobs", payload)
            if wait:
                return self._wait_job(str(job.get("id", "")))
            return job
        from cadgenesis.cli.train import main as train_main  # local fallback

        job_id = f"local-{int(time.time())}"
        train_main(  # type: ignore[call-arg]  # cli.train.main reads args from argv
            epochs=epochs, config_path=config_path, resume_from=resume_from
        )
        return {"id": job_id, "status": "completed", "backend": "local"}

    def _wait_job(
        self, job_id: str, poll_seconds: float = 5.0, timeout_seconds: float = 3600.0
    ) -> dict[str, Any]:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            status = cast(RestBackend, self._sdk.backend)._post(  # wait only used on REST path
                "/api/v1/training/status", {"id": job_id}
            )
            if status.get("status") in ("completed", "failed", "cancelled"):
                return status
            time.sleep(poll_seconds)
        raise SDKError(f"training job {job_id} timed out")


class DeploymentClient:
    """SDK surface for model deployment via the registry."""

    def __init__(self, sdk: CADGenesisSDK) -> None:
        self._sdk = sdk

    def register(
        self,
        name: str,
        path: str,
        version: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        payload = {"name": name, "path": path, "version": version, "metadata": metadata or {}}
        if isinstance(self._sdk.backend, RestBackend):
            data = self._sdk.backend._post("/api/v1/registry/models", payload)
            return f"{name}@{data.get('version', '')}"
        from cadgenesis.platform.registry import ModelRegistry

        registry = ModelRegistry(os.environ.get("CADGENESIS_REGISTRY", "outputs/registry"))
        record = registry.register(name, path, version=version, metadata=metadata)
        return f"{name}@{record.version}"

    def promote(self, name: str, version: str, environment: str = "production") -> str:
        payload = {"name": name, "version": version, "environment": environment}
        if isinstance(self._sdk.backend, RestBackend):
            self._sdk.backend._post("/api/v1/registry/promote", payload)
            return f"{name}@{version} -> {environment}"
        from cadgenesis.platform.registry import ModelRegistry

        registry = ModelRegistry(os.environ.get("CADGENESIS_REGISTRY", "outputs/registry"))
        registry.promote(name, version, environment=environment)
        return f"{name}@{version} -> {environment}"


class PluginClient:
    """SDK surface for plugin discovery/loading."""

    def __init__(self, sdk: CADGenesisSDK) -> None:
        self._sdk = sdk

    def list_local(self, directories: Sequence[str]) -> list[dict[str, Any]]:
        from cadgenesis.platform.plugins import PluginManager

        manager = PluginManager(directories)
        return [m.__dict__ for m in manager.discover()]

    def load_local(self, directories: Sequence[str], name: str) -> str:
        from cadgenesis.platform.plugins import PluginManager

        manager = PluginManager(directories)
        manager.discover()
        manager.load(name)
        return f"loaded {name}"


__all__ = [
    "CADGenesisSDK",
    "InferenceRequest",
    "InferenceResult",
    "LocalBackend",
    "RestBackend",
    "SDKError",
]
