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
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jwt

from fastrag.domain import Principal, Role
from fastrag.errors import AuthenticationError, ConfigurationError

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

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


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def key_id(key_sha256: str) -> str:
    """Short, non-secret identifier for a key, safe to show in listings and audit logs."""
    return key_sha256[:12]


@dataclass(frozen=True)
class ApiKeyRecord:
    key_sha256: str
    principal: Principal
    created_at: float
    expires_at: float | None = None
    revoked_at: float | None = None

    @property
    def key_id(self) -> str:
        return key_id(self.key_sha256)

    def active(self, now: float) -> bool:
        return self.revoked_at is None and (self.expires_at is None or now < self.expires_at)


class ApiKeyStore:
    """Hashed API keys, optionally backed by a JSON file.

    File format: a JSON list of ``{"key_sha256", "tenant_id", "user_id", "groups",
    "roles", "created_at", "expires_at", "revoked_at"}`` (timestamps are Unix seconds).

    The file is re-read when it changes, so keys created or revoked with
    ``fastrag keys ...`` take effect in a running server without a restart. Writers take
    an exclusive lock and re-read before changing anything, so concurrent CLI calls
    don't overwrite each other.
    """

    def __init__(self, path: Path | None = None, clock: Callable[[], float] = time.time):
        self._path = path
        self._clock = clock
        # Readers never block: the (records, file signature) pair is swapped atomically.
        # Writers serialize on _write_lock in-process and on an flock across processes.
        self._state: tuple[dict[str, ApiKeyRecord], tuple[int, int] | None] = ({}, None)
        self._write_lock = threading.Lock()
        self._reload_if_changed()

    # --- reads -------------------------------------------------------------

    def lookup(self, key: str) -> Principal | None:
        self._reload_if_changed()
        record = self._state[0].get(hash_api_key(key))
        if record is None or not record.active(self._clock()):
            return None
        return record.principal

    def records(self) -> list[ApiKeyRecord]:
        self._reload_if_changed()
        return sorted(self._state[0].values(), key=lambda r: r.created_at)

    # --- writes ------------------------------------------------------------

    def create(self, principal: Principal, ttl_seconds: float | None = None) -> str:
        """Register a new key for ``principal`` and return the plaintext (shown once)."""
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        key = generate_api_key()
        now = self._clock()
        record = ApiKeyRecord(
            key_sha256=hash_api_key(key),
            principal=principal,
            created_at=now,
            expires_at=now + ttl_seconds if ttl_seconds is not None else None,
        )
        with self._write() as records:
            records[record.key_sha256] = record
        return key

    def revoke(self, key_id_prefix: str) -> ApiKeyRecord:
        """Revoke the one key whose id starts with ``key_id_prefix``. Idempotent."""
        if len(key_id_prefix) < 6:
            raise ValueError("give at least 6 characters of the key id")
        with self._write() as records:
            matches = [r for h, r in records.items() if h.startswith(key_id_prefix)]
            if len(matches) != 1:
                raise ValueError(f"{len(matches)} keys match {key_id_prefix!r}; need exactly 1")
            (match,) = matches
            if match.revoked_at is not None:
                return match  # already revoked: keep the original revocation time
            revoked = replace(match, revoked_at=self._clock())
            records[revoked.key_sha256] = revoked
        return revoked

    # --- persistence -------------------------------------------------------

    @contextmanager
    def _write(self) -> Iterator[dict[str, ApiKeyRecord]]:
        with self._write_lock:
            if self._path is None:
                records = dict(self._state[0])
                yield records
                self._state = (records, None)
                return
            self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(self._path.with_suffix(".lock"), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                records = self._read_file()  # latest state, including other writers' changes
                yield records
                self._write_file(records)
                self._state = (records, self._file_signature())
            finally:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    def _file_signature(self) -> tuple[int, int] | None:
        if self._path is None:
            return None
        try:
            st = self._path.stat()
        except FileNotFoundError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _reload_if_changed(self) -> None:
        if self._path is None:
            return
        signature = self._file_signature()
        if signature != self._state[1]:
            # Read outside any lock; a concurrent reload just does the same work twice.
            self._state = (self._read_file(), signature)

    def _read_file(self) -> dict[str, ApiKeyRecord]:
        if self._path is None or not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text())
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"API key file {self._path} is not valid JSON") from exc
        if not isinstance(raw, list):
            raise ConfigurationError(f"API key file {self._path} must contain a JSON list")
        return {r["key_sha256"]: self._record(r) for r in raw}

    def _write_file(self, records: dict[str, ApiKeyRecord]) -> None:
        assert self._path is not None
        data = [
            {
                "key_sha256": r.key_sha256,
                "tenant_id": r.principal.tenant_id,
                "user_id": r.principal.user_id,
                "groups": sorted(r.principal.groups),
                "roles": sorted(role.value for role in r.principal.roles),
                "created_at": r.created_at,
                "expires_at": r.expires_at,
                "revoked_at": r.revoked_at,
            }
            for r in sorted(records.values(), key=lambda r: r.created_at)
        ]
        tmp = self._path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(data, indent=2) + "\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(self._path)

    @staticmethod
    def _record(raw: Mapping[str, Any]) -> ApiKeyRecord:
        try:
            return ApiKeyRecord(
                key_sha256=str(raw["key_sha256"]),
                principal=Principal(
                    tenant_id=str(raw["tenant_id"]),
                    user_id=str(raw["user_id"]),
                    groups=_parse_groups(raw.get("groups")),
                    roles=_parse_roles(raw["roles"]),
                ),
                created_at=float(raw.get("created_at") or 0.0),
                expires_at=_optional_float(raw.get("expires_at")),
                revoked_at=_optional_float(raw.get("revoked_at")),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ConfigurationError(f"invalid API key record: {exc}") from exc


@dataclass(frozen=True)
class ClaimMapping:
    """Which token claims carry the tenant, groups and roles.

    Identity providers name these differently (Okta, Entra ID and Keycloak all differ),
    so the mapping is configuration rather than code.
    """

    tenant: str = "tenant"
    groups: str = "groups"
    roles: str = "roles"


DEFAULT_CLAIMS = ClaimMapping()


def _principal_from_claims(claims: Mapping[str, Any], mapping: ClaimMapping) -> Principal:
    tenant = claims.get(mapping.tenant)
    if not isinstance(tenant, str) or not tenant:
        raise ValueError(f"missing or invalid '{mapping.tenant}' claim")
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        raise ValueError("missing or invalid 'sub' claim")
    raw_roles = claims.get(mapping.roles)
    if not isinstance(raw_roles, list):
        raise ValueError(f"'{mapping.roles}' claim must be a list")
    # IdPs often carry roles for other applications too: keep the ones we know, and
    # require at least one, so a token without a fastrag role grants nothing.
    known = {r.value for r in Role}
    roles = frozenset(Role(r) for r in raw_roles if isinstance(r, str) and r in known)
    if not roles:
        raise ValueError("token carries no fastrag role")
    _check_lifetime(claims)
    return Principal(
        tenant_id=tenant,
        user_id=subject,
        groups=_parse_groups(claims.get(mapping.groups)),
        roles=roles,
    )


MAX_TOKEN_LIFETIME_SECONDS = 24 * 3600


def _check_lifetime(claims: Mapping[str, Any]) -> None:
    """Reject tokens issued to live longer than a day: a leaked one stays useful too long."""
    iat, exp = claims.get("iat"), claims.get("exp")
    if not isinstance(iat, int | float) or not isinstance(exp, int | float):
        raise ValueError("'iat' and 'exp' must be numbers")
    if exp - iat > MAX_TOKEN_LIFETIME_SECONDS:
        raise ValueError("token lifetime exceeds 24 hours")


class JwtVerifier:
    """Verifies HS256 tokens signed with a shared secret.

    Suitable for development and for service-to-service tokens you mint yourself. For
    user sign-in, use ``OidcVerifier``: asymmetric keys mean the API can verify tokens
    without being able to mint them.
    """

    ALGORITHMS = ["HS256"]  # pinned: never let the token choose its algorithm

    def __init__(
        self,
        secret: str,
        issuer: str | None = None,
        audience: str | None = None,
        claims: ClaimMapping = DEFAULT_CLAIMS,
    ):
        if len(secret) < 32:
            raise ConfigurationError("JWT secret must be at least 32 characters")
        self._secret = secret
        self._issuer = issuer
        self._audience = audience
        self._claims = claims

    def verify(self, token: str) -> Principal:
        try:
            claims = jwt.decode(
                token,
                self._secret,
                algorithms=self.ALGORITHMS,
                issuer=self._issuer,
                audience=self._audience,
                options={
                    "require": ["exp", "iat", "sub"],
                    "verify_aud": self._audience is not None,
                },
            )
            return _principal_from_claims(claims, self._claims)
        except (jwt.PyJWTError, ValueError) as exc:
            raise AuthenticationError(f"invalid token: {exc}") from exc

    def issue(self, principal: Principal, ttl_seconds: int = 3600, now: float | None = None) -> str:
        """Mint a token. For demos and tests; in production your IdP issues tokens."""
        issued = int(now if now is not None else time.time())
        claims: dict[str, Any] = {
            "sub": principal.user_id,
            self._claims.tenant: principal.tenant_id,
            self._claims.groups: sorted(principal.groups),
            self._claims.roles: sorted(r.value for r in principal.roles),
            "iat": issued,
            "exp": issued + ttl_seconds,
        }
        if self._issuer:
            claims["iss"] = self._issuer
        if self._audience:
            claims["aud"] = self._audience
        return jwt.encode(claims, self._secret, algorithm="HS256")


class OidcVerifier:
    """Verifies tokens from an OpenID Connect provider against its published keys (JWKS).

    Keys are fetched from ``jwks_url`` and cached, and rotated keys are picked up by
    ``kid``. Issuer and audience are mandatory: without them, a token minted by the same
    IdP for a *different* application would be accepted here.
    """

    ALGORITHMS = ("RS256", "ES256", "PS256")

    def __init__(
        self,
        jwks_url: str,
        issuer: str,
        audience: str,
        claims: ClaimMapping = DEFAULT_CLAIMS,
        algorithms: tuple[str, ...] = ALGORITHMS,
        leeway_seconds: int = 30,
        jwk_client: Any | None = None,
    ):
        if not issuer or not audience:
            raise ConfigurationError("OIDC verification requires both issuer and audience")
        if not jwks_url.startswith("https://") and jwk_client is None:
            raise ConfigurationError("the JWKS URL must use https")
        self._jwks = jwk_client or jwt.PyJWKClient(jwks_url, cache_keys=True, lifespan=300)
        self._issuer = issuer
        self._audience = audience
        self._claims = claims
        self._algorithms = list(algorithms)
        self._leeway = leeway_seconds

    def verify(self, token: str) -> Principal:
        try:
            signing_key = self._jwks.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=self._algorithms,
                issuer=self._issuer,
                audience=self._audience,
                leeway=self._leeway,
                options={"require": ["exp", "iat", "sub", "iss", "aud"]},
            )
            return _principal_from_claims(claims, self._claims)
        except (jwt.PyJWTError, ValueError) as exc:
            raise AuthenticationError(f"invalid token: {exc}") from exc


class Authenticator:
    """Resolves request headers to a Principal, or raises ``AuthenticationError``.

    Bearer tokens are routed by their (unverified) ``alg`` header only to choose a
    verifier; each verifier then pins its own algorithms, so the header can't downgrade
    verification. ``alg: none`` matches no verifier and is rejected.
    """

    def __init__(
        self,
        api_keys: ApiKeyStore | None = None,
        jwt_verifier: JwtVerifier | None = None,
        oidc_verifier: OidcVerifier | None = None,
    ):
        if api_keys is None and jwt_verifier is None and oidc_verifier is None:
            raise ConfigurationError("configure at least one authentication method")
        self._api_keys = api_keys
        self._jwt = jwt_verifier
        self._oidc = oidc_verifier

    def authenticate(self, authorization: str | None, api_key: str | None) -> Principal:
        if api_key:
            if self._api_keys is None:
                raise AuthenticationError("API key authentication is not enabled")
            principal = self._api_keys.lookup(api_key)
            if principal is None:
                raise AuthenticationError("invalid, expired or revoked API key")
            return principal
        if authorization:
            scheme, _, token = authorization.partition(" ")
            token = token.strip()
            if scheme.lower() != "bearer" or not token:
                raise AuthenticationError("expected 'Authorization: Bearer <token>'")
            return self._verifier_for(token).verify(token)
        raise AuthenticationError("missing credentials")

    def _verifier_for(self, token: str) -> JwtVerifier | OidcVerifier:
        try:
            alg = jwt.get_unverified_header(token).get("alg", "")
        except jwt.PyJWTError as exc:
            raise AuthenticationError(f"malformed token: {exc}") from exc
        if alg in JwtVerifier.ALGORITHMS and self._jwt is not None:
            return self._jwt
        if alg in OidcVerifier.ALGORITHMS and self._oidc is not None:
            return self._oidc
        raise AuthenticationError(f"unsupported token algorithm {alg!r}")
