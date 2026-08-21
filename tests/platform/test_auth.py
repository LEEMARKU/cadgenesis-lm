from __future__ import annotations

import pytest

from cadgenesis.platform.auth import (
    ABACPolicy,
    Authenticator,
    AuthorizationService,
    InvalidToken,
    Principal,
    RBACPolicy,
    jwt_decode,
    jwt_encode,
)


class TestJWT:
    def test_roundtrip(self):
        token = jwt_encode({"sub": "alice", "roles": ["admin"]}, "secret")
        claims = jwt_decode(token, "secret")
        assert claims["sub"] == "alice"

    def test_wrong_secret(self):
        token = jwt_encode({"sub": "alice"}, "secret")
        with pytest.raises(InvalidToken):
            jwt_decode(token, "wrong")

    def test_expired(self):
        token = jwt_encode({"sub": "alice"}, "secret", expires_in=-10)
        with pytest.raises(InvalidToken):
            jwt_decode(token, "secret")


class TestPrincipal:
    def test_roles_and_projects(self):
        principal = Principal(subject="alice", roles=("admin",), projects={"acme": "editor"})
        assert principal.has_role("admin")
        assert not principal.has_role("user")
        assert principal.has_role("editor", project="acme")
        assert not principal.has_role("editor", project="other")


class TestAuthenticator:
    def setup_method(self):
        self.auth = Authenticator(jwt_secret="test-secret")

    def test_api_key_flow(self):
        _key, raw = self.auth.issue_api_key("service-a", "service-a", roles=("user",))
        principal = self.auth.authenticate_api_key(raw)
        assert principal.subject == "service-a"
        assert principal.has_role("user")

    def test_token_flow(self):
        token = self.auth.issue_token(Principal(subject="bob", roles=("operator",)))
        principal = self.auth.authenticate_token(token)
        assert principal.subject == "bob"
        assert principal.has_role("operator")

    def test_no_credentials(self):
        with pytest.raises(InvalidToken):
            self.auth.authenticate()

    def test_oauth2_password_grant(self):
        token = self.auth.oauth2_password_grant(
            "admin", "admin", verify=lambda u, p: (u, p) == ("admin", "admin")
        )
        assert token is not None
        principal = self.auth.authenticate_token(token)
        assert principal.subject == "admin"

    def test_oauth2_bad_password(self):
        token = self.auth.oauth2_password_grant(
            "admin", "wrong", verify=lambda u, p: (u, p) == ("admin", "admin")
        )
        assert token is None


class TestAuthorization:
    def setup_method(self):
        self.rbac = RBACPolicy(
            {"admin": {"inference:run", "registry:manage"}, "user": {"inference:run"}}
        )
        self.service = AuthorizationService(self.rbac, abac_policies=[])

    def test_can(self):
        admin = Principal(subject="a", roles=("admin",))
        user = Principal(subject="u", roles=("user",))
        assert self.service.can(admin, "registry:manage")
        assert self.service.can(user, "inference:run")
        assert not self.service.can(user, "registry:manage")

    def test_project_scoped_role(self):
        principal = Principal(subject="u", roles=(), projects={"acme": "editor"})
        rbac = RBACPolicy({"editor": {"inference:run"}})
        service = AuthorizationService(rbac)
        assert service.can(principal, "inference:run", project="acme")
        assert not service.can(principal, "inference:run", project="other")

    def test_abac_policy(self):
        policy = ABACPolicy(attribute="tier", op="eq", expected="gold", effect="allow")
        service = AuthorizationService(self.rbac, abac_policies=[policy])
        principal = Principal(subject="u", roles=("user",), attributes={"tier": "gold"})
        assert service.can_abac(principal, "deploy", "cad")

    def test_abac_disabled_allows(self):
        service = AuthorizationService(self.rbac, abac_policies=[])
        principal = Principal(subject="u", roles=("user",))
        assert service.can_abac(principal, "anything", "resource")

    def test_wildcard_permissions(self):
        rbac = RBACPolicy({"admin": {"inference:*", "registry:manage"}, "user": {"inference:run"}})
        service = AuthorizationService(rbac)
        admin = Principal(subject="a", roles=("admin",))
        assert service.can(admin, "inference:run")
        assert service.can(admin, "inference:batch")
        assert service.can(admin, "registry:manage")
        assert not service.can(admin, "training:run")

    def test_import_star_safe(self):
        import importlib

        module = importlib.import_module("cadgenesis.platform.auth")
        names = {name for name in module.__all__}
        for name in names:
            assert hasattr(module, name), name
