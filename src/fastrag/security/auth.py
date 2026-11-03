"""Authentication: turn request credentials into a ``Principal``.

Two mechanisms, both common in enterprise deployments:

* **API keys** for services. Only SHA-256 hashes are stored; the plaintext key is shown
  once at creation. Keys are high-entropy random strings, so a fast hash is appropriate
  (unlike passwords).
* **JWT bearer tokens** for users, as issued by an identity provider. This demo verifies
  HS256 with a shared secret. In production, verify RS256/ES256 against the IdP's JWKS
  (``jwt.PyJWKClient``) - the claim mapping below stays the same.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jwt

from fastrag.domain import Principal, Role
from fastrag.errors import AuthenticationError, ConfigurationError

API_KEY_PREFIX = "frk_"


def generate_api_key() -> str:
    return API_KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _parse_roles(raw: Any) -> frozenset[Role]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("roles must be a non-empty list")
    return frozenset(Role(r) for r in raw)


def _parse_groups(raw: Any) -> frozenset[str]:
    if raw is None:
        return frozenset()
    if not isinstance(raw, list) or not all(isinstance(g, str) for g in raw):
        raise ValueError("groups must be a list of strings")
    return frozenset(raw)


class ApiKeyStore:
    """Hashed API keys, optionally backed by a JSON file.

    File format: a JSON list of ``{"key_sha256", "tenant_id", "user_id", "groups", "roles"}``.
    """

    def __init__(self, path: Path | None = None):
        self._path = path
        self._lock = threading.Lock()
        self._by_hash: dict[str, Principal] = {}
        if path is not None and path.exists():
            for record in json.loads(path.read_text()):
                self._by_hash[record["key_sha256"]] = self._principal(record)

    @staticmethod
    def _principal(record: Mapping[str, Any]) -> Principal:
        try:
            return Principal(
                tenant_id=str(record["tenant_id"]),
                user_id=str(record["user_id"]),
                groups=_parse_groups(record.get("groups")),
                roles=_parse_roles(record["roles"]),
            )
        except (KeyError, ValueError) as exc:
            raise ConfigurationError(f"invalid API key record: {exc}") from exc

    def create(self, principal: Principal) -> str:
        """Register a new key for ``principal`` and return the plaintext (shown once)."""
        key = generate_api_key()
        with self._lock:
            self._by_hash[hash_api_key(key)] = principal
            self._persist()
        return key

    def _persist(self) -> None:
        if self._path is None:
            return
        records = [
            {
                "key_sha256": h,
                "tenant_id": p.tenant_id,
                "user_id": p.user_id,
                "groups": sorted(p.groups),
                "roles": sorted(r.value for r in p.roles),
            }
            for h, p in self._by_hash.items()
        ]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(records, indent=2) + "\n")
        tmp.replace(self._path)

    def lookup(self, key: str) -> Principal | None:
        return self._by_hash.get(hash_api_key(key))


class JwtVerifier:
    """Maps verified JWT claims to a Principal.

    Expected claims: ``sub`` (user id), ``tenant`` (tenant id), ``roles`` (list) and
    optional ``groups`` (list). ``exp`` is required.
    """

    ALGORITHMS = ["HS256"]  # pinned: never let the token choose its algorithm

    def __init__(self, secret: str, issuer: str | None = None, audience: str | None = None):
        if len(secret) < 32:
            raise ConfigurationError("JWT secret must be at least 32 characters")
        self._secret = secret
        self._issuer = issuer
        self._audience = audience

    def verify(self, token: str) -> Principal:
        try:
            claims = jwt.decode(
                token,
                self._secret,
                algorithms=self.ALGORITHMS,
                issuer=self._issuer,
                audience=self._audience,
                options={
                    "require": ["exp", "sub", "tenant"],
                    "verify_aud": self._audience is not None,
                },
            )
            return Principal(
                tenant_id=str(claims["tenant"]),
                user_id=str(claims["sub"]),
                groups=_parse_groups(claims.get("groups")),
                roles=_parse_roles(claims.get("roles")),
            )
        except (jwt.PyJWTError, ValueError) as exc:
            raise AuthenticationError(f"invalid token: {exc}") from exc

    def issue(self, principal: Principal, ttl_seconds: int = 3600, now: float | None = None) -> str:
        """Mint a token. For demos and tests; in production your IdP issues tokens."""
        issued = int(now if now is not None else time.time())
        claims: dict[str, Any] = {
            "sub": principal.user_id,
            "tenant": principal.tenant_id,
            "groups": sorted(principal.groups),
            "roles": sorted(r.value for r in principal.roles),
            "iat": issued,
            "exp": issued + ttl_seconds,
        }
        if self._issuer:
            claims["iss"] = self._issuer
        if self._audience:
            claims["aud"] = self._audience
        return jwt.encode(claims, self._secret, algorithm="HS256")


class Authenticator:
    """Resolves request headers to a Principal, or raises ``AuthenticationError``."""

    def __init__(
        self, api_keys: ApiKeyStore | None = None, jwt_verifier: JwtVerifier | None = None
    ):
        if api_keys is None and jwt_verifier is None:
            raise ConfigurationError("configure at least one authentication method")
        self._api_keys = api_keys
        self._jwt = jwt_verifier

    def authenticate(self, authorization: str | None, api_key: str | None) -> Principal:
        if api_key:
            if self._api_keys is None:
                raise AuthenticationError("API key authentication is not enabled")
            principal = self._api_keys.lookup(api_key)
            if principal is None:
                raise AuthenticationError("invalid API key")
            return principal
        if authorization:
            scheme, _, token = authorization.partition(" ")
            if scheme.lower() != "bearer" or not token:
                raise AuthenticationError("expected 'Authorization: Bearer <token>'")
            if self._jwt is None:
                raise AuthenticationError("token authentication is not enabled")
            return self._jwt.verify(token.strip())
        raise AuthenticationError("missing credentials")
