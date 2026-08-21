"""
cadgenesis.platform.auth
========================
Authentication & authorization for the CADGenesis-LM production platform.

Authentication
    - JWT (HS256/384/512, pure-stdlib) issue/verify/refresh
    - API keys (hashed at rest, scoped to principals)
    - OAuth2 password + client-credentials grants
    - LDAP bind (optional ``ldap3``) and SSO/OIDC (optional discovery via
      ``oauthlib``/``requests``); both degrade to plugin-style adapters

Authorization
    - RBAC: roles -> permission sets
    - ABAC: attribute-based policy evaluation (AND/OR conditions)
    - Project permissions: per-project role scoping

The FastAPI wiring lives in ``cadgenesis.serving.auth_deps``.
"""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import hmac
import json
import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from cadgenesis.platform.security import random_token

logger = logging.getLogger("cadgenesis.platform.auth")

JWT_ALGORITHMS = {"HS256", "HS384", "HS512"}


class AuthError(Exception):
    """Base authentication/authorization failure."""


class InvalidToken(AuthError):
    """Token missing, malformed, expired or tampered."""


class InsufficientPermissions(AuthError):
    """Authenticated but not authorized."""


# --------------------------------------------------------------------- utils


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def sha256_hash(value: str, salt: str | None = None) -> str:
    """Salted SHA-256 hex digest for API-key storage."""
    material = f"{salt or ''}{value}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------- JWT


def jwt_encode(
    payload: Mapping[str, Any],
    secret: str,
    algorithm: str = "HS256",
    expires_in: int = 3600,
    issuer: str | None = None,
    audience: str | None = None,
) -> str:
    """Issue a signed JWT. ``payload`` may include ``sub``, ``roles``, etc."""
    if algorithm not in JWT_ALGORITHMS:
        raise ValueError(f"unsupported JWT algorithm {algorithm!r}")
    now = int(time.time())
    claims: dict[str, Any] = dict(payload)
    claims.setdefault("iat", now)
    claims.setdefault("exp", now + expires_in)
    if issuer:
        claims["iss"] = issuer
    if audience:
        claims["aud"] = audience
    header = {"alg": algorithm, "typ": "JWT"}
    signing_input = (
        _b64url_encode(json.dumps(header, sort_keys=True).encode("utf-8"))
        + "."
        + _b64url_encode(json.dumps(claims, sort_keys=True).encode("utf-8"))
    )
    digest = hmac.new(secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256)
    if algorithm == "HS384":
        digest = hmac.new(secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha384)
    elif algorithm == "HS512":
        digest = hmac.new(secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha512)
    return f"{signing_input}.{_b64url_encode(digest.digest())}"


def jwt_decode(
    token: str,
    secret: str,
    issuer: str | None = None,
    audience: str | None = None,
    leeway: int = 0,
) -> dict[str, Any]:
    """Verify signature + exp/iss/aud and return the claims dict."""
    parts = token.split(".")
    if len(parts) != 3:
        raise InvalidToken("malformed JWT")
    header, body, signature = parts
    try:
        header_obj = json.loads(_b64url_decode(header))
        claims = json.loads(_b64url_decode(body))
    except (ValueError, TypeError) as exc:
        raise InvalidToken("unparseable JWT") from exc
    algorithm = header_obj.get("alg", "HS256")
    if algorithm not in JWT_ALGORITHMS:
        raise InvalidToken(f"unsupported algorithm {algorithm!r}")
    expected = hmac.new(secret.encode("utf-8"), f"{header}.{body}".encode("ascii"), hashlib.sha256)
    if algorithm == "HS384":
        expected = hmac.new(
            secret.encode("utf-8"), f"{header}.{body}".encode("ascii"), hashlib.sha384
        )
    elif algorithm == "HS512":
        expected = hmac.new(
            secret.encode("utf-8"), f"{header}.{body}".encode("ascii"), hashlib.sha512
        )
    if not hmac.compare_digest(signature, _b64url_encode(expected.digest())):
        raise InvalidToken("signature mismatch")
    now = time.time()
    if claims.get("exp", 0) < now - leeway:
        raise InvalidToken("token expired")
    if claims.get("nbf", 0) > now + leeway:
        raise InvalidToken("token not yet valid")
    if issuer and claims.get("iss") != issuer:
        raise InvalidToken("issuer mismatch")
    if audience and claims.get("aud") != audience:
        raise InvalidToken("audience mismatch")
    return claims


# -------------------------------------------------------------- data model


@dataclass(frozen=True)
class Principal:
    """Authenticated identity with its roles and project scoping."""

    subject: str
    roles: tuple[str, ...] = ()
    projects: dict[str, str] = field(default_factory=dict)  # project -> role
    attributes: dict[str, Any] = field(default_factory=dict)

    def has_role(self, role: str, project: str | None = None) -> bool:
        if role in self.roles:
            return True
        return project is not None and self.projects.get(project) == role

    def can(self, permission: str, project: str | None = None) -> bool:
        raise NotImplementedError("use AuthorizationService.can()")


@dataclass
class ApiKey:
    """Stored API key record (never contains the raw key)."""

    name: str
    principal: str
    roles: tuple[str, ...] = ()
    hash: str = ""
    salt: str = ""
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None

    @classmethod
    def create(cls, name: str, principal: str, roles: Sequence[str] = ()) -> tuple[ApiKey, str]:
        """Create a record + the raw key (shown once)."""
        raw = random_token(48)
        salt = random_token(16)
        return cls(
            name=name,
            principal=principal,
            roles=tuple(roles),
            salt=salt,
            hash=sha256_hash(raw, salt),
        ), raw


# ---------------------------------------------------------- authorization


class RBACPolicy:
    """Role -> permission mapping."""

    def __init__(self, role_permissions: Mapping[str, Iterable[str]] | None = None) -> None:
        self._map: dict[str, set[str]] = {k: set(v) for k, v in (role_permissions or {}).items()}

    def add_role(self, role: str, permissions: Iterable[str]) -> None:
        self._map.setdefault(role, set()).update(permissions)

    def permissions_for(self, roles: Iterable[str]) -> set[str]:
        granted: set[str] = set()
        for role in roles:
            granted |= self._map.get(role, set())
        return granted

    def permits(self, roles: Iterable[str], permission: str) -> bool:
        granted = self.permissions_for(roles)
        if permission in granted:
            return True
        return any(fnmatch.fnmatchcase(permission, pattern) for pattern in granted)


@dataclass(frozen=True)
class ABACPolicy:
    """Attribute condition: ``attribute`` op ``expected``; ops: eq/neq/in/gt/lt."""

    attribute: str
    op: str = "eq"
    expected: Any = True
    effect: str = "allow"

    def evaluate(self, attributes: Mapping[str, Any]) -> bool:
        value = attributes.get(self.attribute)
        if self.op == "eq":
            return value == self.expected
        if self.op == "neq":
            return value != self.expected
        if self.op == "in":
            return value in self.expected
        if self.op == "gt":
            return value is not None and value > self.expected
        if self.op == "lt":
            return value is not None and value < self.expected
        raise ValueError(f"unknown ABAC op {self.op!r}")


class AuthorizationService:
    """RBAC + ABAC + project-scoped permission evaluation."""

    def __init__(
        self,
        rbac: RBACPolicy | None = None,
        abac_policies: Sequence[ABACPolicy] | None = None,
        default_permission: str = "cadgenesis:*:read",
    ) -> None:
        self.rbac = rbac or RBACPolicy()
        self.abac_policies = list(abac_policies or [])
        self.default_permission = default_permission

    def can(
        self,
        principal: Principal,
        permission: str,
        project: str | None = None,
    ) -> bool:
        roles = set(principal.roles)
        if project and principal.projects.get(project):
            roles.add(principal.projects[project])
        if self.rbac.permits(roles, permission):
            return True
        if permission == self.default_permission and not roles:
            return False
        return False

    def can_abac(self, principal: Principal, action: str, resource: str) -> bool:
        """Evaluate ABAC policies against the principal's attributes."""
        if not self.abac_policies:
            return True  # ABAC disabled = allow
        context = {"action": action, "resource": resource, **principal.attributes}
        decisions = [p.evaluate(context) for p in self.abac_policies]
        return all(
            p.effect == "allow" for p, ok in zip(self.abac_policies, decisions, strict=True) if ok
        ) and any(decisions)


# ----------------------------------------------------------- authenticator


class Authenticator:
    """Composite authenticator: JWT + API keys + OAuth2 + LDAP + SSO providers."""

    def __init__(
        self, jwt_secret: str, issuer: str | None = None, audience: str | None = None
    ) -> None:
        self.jwt_secret = jwt_secret
        self.issuer = issuer
        self.audience = audience
        self._api_keys: dict[str, ApiKey] = {}
        self._principals: dict[str, Principal] = {}

    # API keys --------------------------------------------------------------

    def issue_api_key(
        self, name: str, principal: str, roles: Sequence[str] = ()
    ) -> tuple[ApiKey, str]:
        record, raw = ApiKey.create(name, principal, roles)
        self._api_keys[record.name] = record
        return record, raw

    def authenticate_api_key(self, raw_key: str) -> Principal:
        for record in self._api_keys.values():
            if hmac.compare_digest(sha256_hash(raw_key, record.salt), record.hash):
                if record.expires_at is not None and record.expires_at < time.time():
                    raise InvalidToken("API key expired")
                return Principal(subject=record.principal, roles=record.roles)
        raise InvalidToken("invalid API key")

    # JWT -------------------------------------------------------------------

    def issue_token(self, principal: Principal, expires_in: int = 3600) -> str:
        return jwt_encode(
            {
                "sub": principal.subject,
                "roles": list(principal.roles),
                "projects": principal.projects,
            },
            self.jwt_secret,
            issuer=self.issuer,
            audience=self.audience,
            expires_in=expires_in,
        )

    def authenticate_token(self, token: str) -> Principal:
        claims = jwt_decode(token, self.jwt_secret, issuer=self.issuer, audience=self.audience)
        return Principal(
            subject=str(claims["sub"]),
            roles=tuple(claims.get("roles", [])),
            projects=dict(claims.get("projects", {})),
        )

    # OAuth2 (server-side; use with FastAPI OAuth2PasswordRequestForm) -------

    def oauth2_password_grant(
        self,
        username: str,
        password: str,
        verify: Callable[[str, str], bool],
        principal: Principal | None = None,
    ) -> str | None:
        if not verify(username, password):
            return None
        p = principal or self._principals.get(
            username, Principal(subject=username, roles=("user",))
        )
        return self.issue_token(p)

    # LDAP ------------------------------------------------------------------

    def authenticate_ldap(
        self,
        server: str,
        username: str,
        password: str,
        base_dn: str,
        ldap_connector: Callable[..., Any] | None = None,
    ) -> Principal | None:
        """LDAP bind; use ``ldap3`` connector when installed."""
        if ldap_connector is not None:
            if ldap_connector(server, username, password, base_dn):
                return Principal(subject=username, roles=("user",))
            return None
        try:  # pragma: no cover - requires ldap3
            import ldap3  # type: ignore[import-not-found]

            server_obj = ldap3.Server(server)
            with ldap3.Connection(server_obj, user=username, password=password, auto_bind=True):
                return Principal(subject=username, roles=("user",))
        except ImportError:
            raise RuntimeError(
                "LDAP authentication requires 'ldap3' or a custom connector"
            ) from None
        except ldap3.core.exceptions.LDAPException:
            return None

    # SSO / OIDC ------------------------------------------------------------

    def authenticate_sso(
        self,
        id_token: str,
        jwks_secret: str | None = None,
        verify: Callable[[dict[str, Any]], Principal] | None = None,
    ) -> Principal:
        """OIDC-style SSO: verify the ID token (HS* via jwks secret or custom)."""
        if verify is not None:
            return verify({"token": id_token})
        claims = jwt_decode(id_token, jwks_secret or self.jwt_secret, issuer=self.issuer)
        return Principal(subject=str(claims["sub"]), roles=tuple(claims.get("roles", [])))

    def authenticate(self, token: str | None = None, api_key: str | None = None) -> Principal:
        """Ordered attempt: JWT bearer -> API key."""
        if token:
            return self.authenticate_token(token)
        if api_key:
            return self.authenticate_api_key(api_key)
        raise InvalidToken("no credentials provided")


__all__ = [
    "JWT_ALGORITHMS",
    "ABACPolicy",
    "AuthError",
    "Authenticator",
    "AuthorizationService",
    "InsufficientPermissions",
    "InvalidToken",
    "Principal",
    "RBACPolicy",
    "jwt_decode",
    "jwt_encode",
]
