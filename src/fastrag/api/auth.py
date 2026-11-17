"""Service-token authentication (D9).

Tokens look like `fr_<prefix>_<secret>` and are stored as SHA-256 hashes. Each
key belongs to one tenant and carries a ceiling of access groups: a key
issued for the customer agent can only ever read `public` content.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from fastrag.config import get_settings
from fastrag.db.engine import get_sessionmaker
from fastrag.db.models import ApiKey, Tenant


@dataclass(frozen=True)
class Principal:
    tenant_id: uuid.UUID | None  # None: operator admin, must name a tenant per request
    allowed_groups: tuple[str, ...]
    allowed_profiles: tuple[str, ...] | None
    is_admin: bool
    is_operator: bool = False
    key_id: uuid.UUID | None = None


def new_token() -> tuple[str, str, str]:
    """Return (token, prefix, hash). The token is shown once."""
    prefix = secrets.token_hex(4)
    token = f"fr_{prefix}_{secrets.token_urlsafe(32)}"
    return token, prefix, hash_token(token)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# Per-process cache. A revoked key stops working at once in the process that
# revoked it and within the TTL in other API processes.
_CACHE: dict[str, tuple[float, Principal | None]] = {}
_CACHE_TTL_S = 30.0


def clear_auth_cache() -> None:
    _CACHE.clear()


async def _lookup(token: str) -> Principal | None:
    h = hash_token(token)
    hit = _CACHE.get(h)
    if hit and hit[0] > time.monotonic():
        return hit[1]
    async with get_sessionmaker()() as session:
        key = (
            await session.execute(select(ApiKey).where(ApiKey.key_hash == h, ApiKey.revoked_at.is_(None)))
        ).scalar_one_or_none()
        principal = None
        if key is not None:
            principal = Principal(
                tenant_id=key.tenant_id,
                allowed_groups=tuple(key.allowed_groups),
                allowed_profiles=tuple(key.allowed_profiles) if key.allowed_profiles else None,
                is_admin=key.is_admin,
                key_id=key.id,
            )
            await session.execute(update(ApiKey).where(ApiKey.id == key.id).values(last_used_at=datetime.now(UTC)))
            await session.commit()
    _CACHE[h] = (time.monotonic() + _CACHE_TTL_S, principal)
    return principal


async def get_principal(authorization: str | None = Header(default=None)) -> Principal:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token", headers={"WWW-Authenticate": "Bearer"})
    token = authorization[7:].strip()
    admin = get_settings().admin_token.get_secret_value()
    if admin and hmac.compare_digest(token.encode(), admin.encode()):
        return Principal(None, ("public", "staff"), None, is_admin=True, is_operator=True)
    principal = await _lookup(token)
    if principal is None:
        raise HTTPException(401, "invalid or revoked token", headers={"WWW-Authenticate": "Bearer"})
    return principal


async def require_admin(principal: Principal = Depends(get_principal)) -> Principal:
    if not principal.is_admin:
        raise HTTPException(403, "admin token required")
    return principal


async def require_operator(principal: Principal = Depends(get_principal)) -> Principal:
    if not principal.is_operator:
        raise HTTPException(403, "operator admin token required")
    return principal


async def resolve_tenant(session: AsyncSession, principal: Principal, tenant_header: str | None) -> uuid.UUID:
    """Tenant keys are bound to their tenant; the operator token names one
    with the X-Tenant header (slug)."""
    if principal.tenant_id is not None:
        return principal.tenant_id
    if not tenant_header:
        raise HTTPException(400, "X-Tenant header is required with the operator token")
    tid = (await session.execute(select(Tenant.id).where(Tenant.slug == tenant_header))).scalar_one_or_none()
    if tid is None:
        raise HTTPException(404, f"tenant {tenant_header!r} not found")
    return tid


def trace_id(request: Request) -> str:
    return str(getattr(request.state, "trace_id", "") or uuid.uuid4().hex)
