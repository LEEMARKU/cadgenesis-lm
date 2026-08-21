"""
cadgenesis.platform
===================
Production platform services for CADGenesis-LM v6.0 (Pillar 11).

- config: multi-format configuration (JSON/YAML/TOML/env) with reload
- security: secrets, AES-GCM encryption, audit logging
- auth: JWT, API keys, OAuth2, LDAP, SSO + RBAC/ABAC authorization
- plugins: runtime plugin loading with dependency/version validation
- registry: versioned model registry with rollback & deployment history
- monitoring: Prometheus export, OpenTelemetry bridge, health aggregation
- logging: distributed logging and log aggregation
- sdk: high-level Python SDK (inference/training/deployment/plugins)
- dashboard: operational dashboard generation
"""

from cadgenesis.platform.auth import (
    ABACPolicy,
    Authenticator,
    AuthError,
    AuthorizationService,
    InsufficientPermissions,
    InvalidToken,
    Principal,
    RBACPolicy,
    jwt_decode,
    jwt_encode,
)
from cadgenesis.platform.config import ConfigStore, load_config
from cadgenesis.platform.logging import DistributedLogClient, LogAggregator
from cadgenesis.platform.monitoring import (
    HealthAggregator,
    OpenTelemetryBridge,
    PrometheusExporter,
    grafana_dashboard,
    render_prometheus,
)
from cadgenesis.platform.plugins import PlatformPlugin, PluginError, PluginManager, PluginManifest
from cadgenesis.platform.registry import DeploymentRecord, ModelRegistry, ModelVersion
from cadgenesis.platform.sdk import CADGenesisSDK, InferenceRequest, InferenceResult, SDKError
from cadgenesis.platform.security import AuditLogger, CryptoService, SecretManager, random_token

__all__ = [
    "ABACPolicy",
    "AuditLogger",
    "AuthError",
    "Authenticator",
    "AuthorizationService",
    "CADGenesisSDK",
    "ConfigStore",
    "CryptoService",
    "DeploymentRecord",
    "DistributedLogClient",
    "HealthAggregator",
    "InferenceRequest",
    "InferenceResult",
    "InsufficientPermissions",
    "InvalidToken",
    "LogAggregator",
    "ModelRegistry",
    "ModelVersion",
    "OpenTelemetryBridge",
    "PlatformPlugin",
    "PluginError",
    "PluginManager",
    "PluginManifest",
    "Principal",
    "PrometheusExporter",
    "RBACPolicy",
    "SDKError",
    "SecretManager",
    "grafana_dashboard",
    "jwt_decode",
    "jwt_encode",
    "load_config",
    "random_token",
    "render_prometheus",
]
