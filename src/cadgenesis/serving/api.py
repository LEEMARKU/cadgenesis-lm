"""
cadgenesis.serving.api
======================
Production REST API for the CADGenesis-LM platform.

- Versioned endpoints under ``/api/v1`` (inference, training, registry, auth)
- Async request handling (engine calls run in a thread pool)
- SSE streaming responses for token-level generation
- OpenAPI + Swagger (``/docs``, ``/redoc``, ``/openapi.json``)
- OAuth2/JWT/API-key authentication with RBAC authorization
- Prometheus ``/metrics`` and health/ready probes
- WebSocket endpoint at ``/ws`` (see ``cadgenesis.serving.websocket``)

Requires ``fastapi`` + ``uvicorn`` (optional dependencies; ``app`` is ``None``
when FastAPI is unavailable).  Run with::

    uvicorn cadgenesis.serving.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import torch

try:
    from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
    from fastapi.security import (
        APIKeyHeader,
        HTTPAuthorizationCredentials,
        HTTPBearer,
        OAuth2PasswordRequestForm,
    )
    from pydantic import BaseModel, Field
    from starlette.concurrency import run_in_threadpool
except ImportError:  # pragma: no cover - optional dependency
    FastAPI = None  # type: ignore[assignment]
    app = None  # type: ignore[assignment]

from cadgenesis import __version__ as CADGENESIS_VERSION
from cadgenesis.monitoring.health import HealthChecker, check_disk_usage, check_memory_usage
from cadgenesis.platform.auth import (
    Authenticator,
    AuthorizationService,
    InvalidToken,
    Principal,
    RBACPolicy,
)
from cadgenesis.platform.config import ConfigStore, load_config
from cadgenesis.platform.monitoring import HealthAggregator, PrometheusExporter
from cadgenesis.platform.registry import ModelRegistry
from cadgenesis.platform.security import AuditLogger
from cadgenesis.serving.batching import DynamicBatcher
from cadgenesis.serving.lifecycle import ModelLifecycle, resolve_registry_path
from cadgenesis.serving.websocket import websocket_endpoint
from cadgenesis.telemetry.metrics import MetricsRegistry

logger = logging.getLogger("cadgenesis.serving.api")


class GenerateRequest(BaseModel):
    text: str = Field(..., description="Natural-language design request")
    max_len: int = Field(64, ge=1, le=2048)
    beam_width: int = Field(1, ge=1, le=16)
    temperature: float = Field(1.0, ge=0.0)


class StreamRequest(BaseModel):
    text: str = Field(..., description="Natural-language design request")
    max_len: int = Field(64, ge=1, le=2048)


class RegisterRequest(BaseModel):
    name: str
    path: str
    version: str | None = None
    metadata: dict[str, Any] = {}


class PromoteRequest(BaseModel):
    name: str
    version: str
    environment: str = "production"


class ApiKeyRequest(BaseModel):
    name: str
    principal: str = "user"
    roles: list[str] = ["user"]


class _ServeState:
    """Mutable serving state shared across handlers (FastAPI 2.x friendly)."""

    def __init__(self) -> None:
        self.lifecycle: ModelLifecycle | None = None
        self.batcher: DynamicBatcher | None = None
        self.registry: ModelRegistry | None = None
        self.authenticator: Authenticator | None = None
        self.authorizer: AuthorizationService | None = None
        self.audit: AuditLogger | None = None
        self.config: ConfigStore | None = None


def _build_metrics(registry: MetricsRegistry) -> dict[str, Any]:
    """Register the platform metric families; returns name -> metric handles."""
    registry.clear()
    return {
        "requests": registry.counter("inference_requests", "Inference requests processed"),
        "errors": registry.counter("inference_errors", "Failed inference requests"),
        "latency": registry.histogram(
            "inference_latency",
            "Inference latency (seconds)",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, float("inf")],
        ),
        "loads": registry.counter("model_loads", "Model loads performed"),
        "active": registry.gauge("active_models", "Models currently loaded"),
    }


def _default_load_fn(path: str) -> Any:
    """Load a checkpoint into a CADInferenceEngine (default model factory)."""

    from cadgenesis.config import CADConfig
    from cadgenesis.inference.engine import CADInferenceEngine
    from cadgenesis.tokenizer import AutonomousCADTokenizer, restore_vocab_tokens
    from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = (
        CADConfig.from_dict(checkpoint["config"])
        if isinstance(checkpoint.get("config"), dict)
        else CADConfig.mini()
    )
    device = os.environ.get("CADGENESIS_DEVICE", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_dir = os.environ.get("CADGENESIS_TOKENIZER")
    if tokenizer_dir and os.path.isdir(tokenizer_dir):
        tokenizer = AutonomousCADTokenizer.load(tokenizer_dir)
    else:
        tokenizer = AutonomousCADTokenizer.build_mini()
    restore_vocab_tokens(tokenizer, checkpoint.get("vocab_tokens", []))
    model = GeometryAwareTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return CADInferenceEngine(model, tokenizer, device=device)


def _default_warmup(engine: Any) -> None:
    try:
        engine.greedy("box", max_len=8)
    except Exception as exc:
        logger.warning("model warm-up failed (continuing): %s", exc)


def _build_auth(config: ConfigStore) -> tuple[Authenticator, AuthorizationService]:
    secret = (
        os.environ.get("CADGENESIS_JWT_SECRET")
        or config.get("auth.jwt_secret")
        or "dev-only-secret-change-me"
    )
    issuer = config.get("auth.issuer") or "cadgenesis"
    authenticator = Authenticator(secret, issuer=issuer)
    rbac = RBACPolicy(
        {
            "admin": ["inference:*", "training:*", "registry:*", "auth:manage"],
            "operator": ["inference:*", "training:run", "registry:read"],
            "user": ["inference:run", "registry:read"],
        }
    )
    authorizer = AuthorizationService(rbac=rbac)
    return authenticator, authorizer


def _audit_path(config: ConfigStore | None) -> str:
    configured = config.get("audit.path") if config else None
    return str(configured or os.environ.get("CADGENESIS_AUDIT_LOG", "outputs/audit.jsonl"))


def create_app(
    config: ConfigStore | None = None,
    lifecycle: ModelLifecycle | None = None,
    engine: Any | None = None,
) -> Any:
    """Build the FastAPI application (returns ``None`` without FastAPI)."""
    if FastAPI is None:
        return None

    state = _ServeState()
    state.config = config or load_config(
        [os.environ.get("CADGENESIS_CONFIG", "configs/platform.json")]
        if os.path.exists(os.environ.get("CADGENESIS_CONFIG", "configs/platform.json"))
        else None
    )
    state.authenticator, state.authorizer = _build_auth(state.config)
    state.audit = AuditLogger(_audit_path(state.config))
    state.registry = ModelRegistry(str(state.config.get("registry.directory", "outputs/registry")))
    if lifecycle is not None:
        state.lifecycle = lifecycle
    elif engine is not None:

        class _OneEngineLifecycle:
            def load(self, name: str, path: str, version: str | None = None, force: bool = False):
                return self

            def engine(self, name: str = "default"):
                return engine

            def names(self) -> list[str]:
                return ["default"]

            def status(self) -> list[dict[str, Any]]:
                return [{"name": "default", "path": "injected", "healthy": True}]

            def __contains__(self, name: str) -> bool:
                return name == "default"

        state.lifecycle = _OneEngineLifecycle()  # type: ignore[assignment]
    else:
        state.lifecycle = ModelLifecycle(_default_load_fn, _default_warmup)

    config = state.config
    lifecycle = state.lifecycle
    registry = state.registry
    if config is None or lifecycle is None or registry is None:
        raise RuntimeError("serving state not initialized")

    batcher = DynamicBatcher(
        dispatch=lambda payloads: [
            lifecycle.engine().greedy(p["text"], max_len=int(p.get("max_len", 128)))
            for p in payloads
        ],
        max_batch=int(config.get("serving.max_batch", 8)),
        max_wait_seconds=float(config.get("serving.max_wait_seconds", 0.02)),
    )
    state.batcher = batcher
    metric_registry = MetricsRegistry(prefix="cadgenesis")
    metrics = _build_metrics(metric_registry)

    @asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[None]:
        model_path = os.environ.get("CADGENESIS_MODEL") or config.get("serving.model_path")
        registry_path = resolve_registry_path(
            str(config.get("registry.directory")), "cadgenesis", "production"
        )
        path = model_path or registry_path
        if path:
            try:
                await run_in_threadpool(lifecycle.load, "default", path)
                metrics["loads"].inc()
            except Exception:
                logger.exception("model preload failed; serving without a model")
        metrics["active"].set(len(lifecycle.names()))
        logger.info("CADGenesis serving API ready (v%s)", CADGENESIS_VERSION)
        yield
        batcher.shutdown()

    app = FastAPI(
        title="CADGenesis-LM Platform API",
        version=CADGENESIS_VERSION,
        description=(
            "Production platform API for CADGenesis-LM v6.0: inference, training, "
            "model registry, authentication and observability."
        ),
        lifespan=lifespan,
        openapi_tags=[
            {
                "name": "inference",
                "description": "CAD sequence generation (sync, streaming, batch)",
            },
            {"name": "training", "description": "Training job control"},
            {"name": "registry", "description": "Model registry: versions, promotion, rollback"},
            {"name": "auth", "description": "Authentication: tokens, API keys"},
            {"name": "ops", "description": "Health, readiness, metrics, version"},
        ],
    )
    app.state.serving = state
    app.state.metrics = metrics
    bearer = HTTPBearer(auto_error=False)
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    # ------------------------------------------------------------ auth deps

    def resolve_principal(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
        api_key: str | None = Depends(api_key_header),
    ) -> Principal | None:
        if state.authenticator is None:
            return None
        if credentials and credentials.credentials:
            try:
                return state.authenticator.authenticate_token(credentials.credentials)
            except InvalidToken as exc:
                raise HTTPException(status_code=401, detail=str(exc)) from exc
        if api_key:
            try:
                return state.authenticator.authenticate_api_key(api_key)
            except InvalidToken as exc:
                raise HTTPException(status_code=401, detail=str(exc)) from exc
        return None

    def require_permission(permission: str):
        def checker(principal: Principal | None = Depends(resolve_principal)) -> Principal | None:
            if state.authorizer is None:
                return principal
            if principal is None or not state.authorizer.can(principal, permission):
                raise HTTPException(status_code=403, detail=f"permission required: {permission}")
            return principal

        return checker

    # ------------------------------------------------------------- payloads

    # -------------------------------------------------------------- helpers

    def _run_inference(payload: Mapping[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        engine = lifecycle.engine("default")
        try:
            if int(payload.get("beam_width", 1)) > 1:
                result = engine.beam(
                    str(payload["text"]),
                    beam_width=int(payload["beam_width"]),
                    max_len=int(payload.get("max_len", 64)),
                )
            else:
                result = engine.greedy(
                    str(payload["text"]), max_len=int(payload.get("max_len", 64))
                )
            latency = time.perf_counter() - start
            metrics["requests"].inc()
            metrics["latency"].observe(latency)
            return {
                "text": " ".join(result.tokens),
                "tokens": result.tokens,
                "confidence": round(float(result.confidence), 6),
                "toon": result.toon,
                "latency_ms": round(latency * 1000.0, 3),
            }
        except Exception as exc:
            metrics["errors"].inc()
            raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

    # ----------------------------------------------------------------- ops

    @app.get("/healthz", tags=["ops"], summary="Liveness probe")
    def healthz() -> dict[str, Any]:
        from cadgenesis.monitoring.health import HealthResult

        checker = HealthChecker()
        checker.register("memory", check_memory_usage)
        checker.register("disk", lambda: check_disk_usage("."))
        if state.lifecycle:
            names = len(state.lifecycle.names())
            checker.register(
                "models", lambda: HealthResult("models", ok=True, detail=f"{names} loaded")
            )
        return HealthAggregator(checker).summary()

    @app.get("/readyz", tags=["ops"], summary="Readiness probe")
    def readyz() -> dict[str, Any]:
        names = state.lifecycle.names() if state.lifecycle else []
        ready = len(names) > 0
        status = 200 if ready else 503
        return JSONResponse(status_code=status, content={"ready": ready, "models": names})

    @app.get(
        "/metrics", tags=["ops"], summary="Prometheus metrics", response_class=PlainTextResponse
    )
    def metrics_endpoint() -> str:
        return PrometheusExporter(metric_registry).render()

    @app.get("/api/v1/version", tags=["ops"], summary="Platform version")
    def version() -> dict[str, str]:
        return {"version": CADGENESIS_VERSION, "api": "v1"}

    @app.post("/api/v1/config/reload", tags=["ops"], summary="Dynamic configuration reload")
    async def config_reload() -> dict[str, Any]:
        changed = await run_in_threadpool(config.reload)
        return {"reloaded": True, "changed": changed, "sources": len(config.sources)}

    @app.get("/api/v1/models", tags=["ops"], summary="Loaded models")
    def loaded_models() -> list[dict[str, Any]]:
        return state.lifecycle.status() if state.lifecycle else []

    # ------------------------------------------------------------- inference

    @app.post(
        "/api/v1/inference/generate", tags=["inference"], summary="Generate CAD tokens (sync)"
    )
    async def generate(
        request: GenerateRequest,
        principal: Principal | None = Depends(require_permission("inference:run")),
    ) -> dict[str, Any]:
        if state.lifecycle is None or "default" not in state.lifecycle:
            raise HTTPException(status_code=503, detail="no model loaded")
        return await run_in_threadpool(_run_inference, request.model_dump())

    @app.post("/api/v1/inference/batch", tags=["inference"], summary="Batch generation")
    async def generate_batch(
        request: Request,
        principal: Principal | None = Depends(require_permission("inference:run")),
    ) -> list[dict[str, Any]]:
        payload = await request.json()
        texts = payload.get("texts", [])
        if not isinstance(texts, list) or not texts:
            raise HTTPException(status_code=422, detail="'texts' must be a non-empty list")
        engine = lifecycle.engine("default")
        results = await run_in_threadpool(
            engine.batch_generate, texts, max_len=int(payload.get("max_len", 64))
        )
        return [
            {
                "text": " ".join(r.tokens),
                "tokens": r.tokens,
                "confidence": round(float(r.confidence), 6),
                "toon": r.toon,
            }
            for r in results
        ]

    @app.post("/api/v1/inference/stream", tags=["inference"], summary="Streaming generation (SSE)")
    async def stream(request: StreamRequest) -> StreamingResponse:
        async def event_source() -> AsyncIterator[str]:
            try:
                engine = lifecycle.engine("default")
                result = await run_in_threadpool(
                    engine.greedy, request.text, max_len=request.max_len
                )
                for index, token in enumerate(result.tokens):
                    payload = json.dumps({"index": index, "token": token, "done": False})
                    yield f"data: {payload}\n\n"
                    await asyncio.sleep(0)
                final = json.dumps(
                    {
                        "done": True,
                        "text": " ".join(result.tokens),
                        "confidence": round(float(result.confidence), 6),
                        "toon": result.toon,
                    }
                )
                yield f"data: {final}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'done': True, 'error': str(exc)})}\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    # ------------------------------------------------------------- training

    @app.post("/api/v1/training/jobs", tags=["training"], summary="Launch a training job")
    async def launch_training(
        request: Request,
        principal: Principal | None = Depends(require_permission("training:run")),
    ) -> dict[str, Any]:
        payload = await request.json()
        job_id = f"job-{int(time.time())}"
        return {
            "id": job_id,
            "status": "queued",
            "config": payload.get("config"),
            "epochs": payload.get("epochs"),
        }

    @app.post("/api/v1/training/status", tags=["training"], summary="Training job status")
    async def training_status(request: Request) -> dict[str, Any]:
        payload = await request.json()
        return {
            "id": payload.get("id"),
            "status": "unknown",
            "detail": "remote job tracking requires a scheduler backend",
        }

    # ------------------------------------------------------------- registry

    @app.post("/api/v1/registry/models", tags=["registry"], summary="Register a model version")
    async def register_model(
        request: RegisterRequest,
        principal: Principal | None = Depends(require_permission("registry:write")),
    ) -> dict[str, Any]:
        record = await run_in_threadpool(
            registry.register, request.name, request.path, request.version, request.metadata
        )
        metrics["loads"].inc()
        return {"name": record.name, "version": record.version, "path": record.path}

    @app.get("/api/v1/registry/models/{name}", tags=["registry"], summary="List model versions")
    def registry_versions(
        name: str,
        principal: Principal | None = Depends(require_permission("registry:read")),
    ) -> list[dict[str, Any]]:
        return [v.to_dict() for v in registry.list_versions(name)]

    @app.post(
        "/api/v1/registry/promote", tags=["registry"], summary="Promote a version to an environment"
    )
    def promote_model(
        request: PromoteRequest,
        principal: Principal | None = Depends(require_permission("registry:write")),
    ) -> dict[str, Any]:
        record = registry.promote(request.name, request.version, request.environment)
        return {"name": record.name, "version": record.version, "environment": request.environment}

    @app.post("/api/v1/registry/rollback", tags=["registry"], summary="Roll back an environment")
    def rollback_model(
        request: PromoteRequest,
        principal: Principal | None = Depends(require_permission("registry:write")),
    ) -> dict[str, Any]:
        record = registry.rollback(request.name, request.environment)
        if record is None:
            raise HTTPException(status_code=404, detail="no previous deployment to roll back to")
        return {"name": record.name, "version": record.version, "environment": request.environment}

    # ---------------------------------------------------------------- auth

    @app.post("/api/v1/auth/token", tags=["auth"], summary="OAuth2 password grant -> JWT")
    async def oauth_token(form: OAuth2PasswordRequestForm = Depends()) -> dict[str, Any]:
        if state.authenticator is None:
            raise HTTPException(status_code=503, detail="auth disabled")
        token = state.authenticator.oauth2_password_grant(
            form.username,
            form.password,
            verify=lambda user, pwd: (
                user == os.environ.get("CADGENESIS_ADMIN_USER", "admin")
                and pwd == os.environ.get("CADGENESIS_ADMIN_PASSWORD", "admin")
            ),
            principal=Principal(subject=form.username, roles=("admin",)),
        )
        if token is None:
            if state.audit:
                state.audit.log(form.username, "login", "auth/token", "denied", severity="warning")
            raise HTTPException(status_code=401, detail="invalid credentials")
        if state.audit:
            state.audit.log(form.username, "login", "auth/token", "success")
        return {"access_token": token, "token_type": "bearer"}

    @app.post("/api/v1/auth/api-keys", tags=["auth"], summary="Issue an API key")
    async def issue_api_key(
        request: ApiKeyRequest,
        principal: Principal | None = Depends(require_permission("auth:manage")),
    ) -> dict[str, str]:
        if state.authenticator is None:
            raise HTTPException(status_code=503, detail="auth disabled")
        _, raw = state.authenticator.issue_api_key(request.name, request.principal, request.roles)
        if state.audit:
            state.audit.log(
                str(principal.subject if principal else "?"),
                "issue_api_key",
                request.name,
                "success",
            )
        return {"api_key": raw, "name": request.name}

    @app.get("/api/v1/auth/me", tags=["auth"], summary="Current principal")
    def whoami(principal: Principal | None = Depends(resolve_principal)) -> dict[str, Any]:
        if principal is None:
            raise HTTPException(status_code=401, detail="unauthenticated")
        return {
            "subject": principal.subject,
            "roles": list(principal.roles),
            "projects": principal.projects,
        }

    # ------------------------------------------------------------ websocket

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket_endpoint(websocket, state)

    return app


app = create_app() if FastAPI is not None else None


def get_app() -> Any:
    """Import-safe accessor: builds the app on first call."""
    if app is None:
        return create_app()
    return app


__all__ = ["app", "create_app", "get_app"]
