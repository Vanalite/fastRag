from __future__ import annotations

import uuid

from fastapi import HTTPException
from sqlalchemy import not_, select
from sqlalchemy.ext.asyncio import AsyncSession

from fastrag.api.auth import Principal
from fastrag.db.models import Collection
from fastrag.retrieval.profiles import Profile, effective_groups, resolve_profile


async def collections_for(
    session: AsyncSession, tenant_id: uuid.UUID, slugs: list[str] | None, ctype: str
) -> list[Collection]:
    stmt = select(Collection).where(Collection.tenant_id == tenant_id, Collection.type == ctype)
    if slugs:
        stmt = stmt.where(Collection.slug.in_(slugs))
    else:
        # Trial collections built by `fastrag compare` are only searched by name.
        stmt = stmt.where(not_(Collection.settings.has_key("trial_of")))
    found = list((await session.execute(stmt.order_by(Collection.slug))).scalars())
    if slugs:
        missing = sorted(set(slugs) - {c.slug for c in found})
        if missing:
            raise HTTPException(404, f"{ctype} collection(s) not found: {', '.join(missing)}")
    return found


def profile_and_groups(
    principal: Principal, profile_name: str, requested: list[str] | None, collections: list[Collection]
) -> tuple[Profile, list[str]]:
    if principal.allowed_profiles is not None and profile_name not in principal.allowed_profiles:
        raise HTTPException(403, f"this key may not use the {profile_name!r} profile")
    settings = collections[0].settings if collections else None
    try:
        profile = resolve_profile(profile_name, settings)
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc
    groups = effective_groups(requested, list(principal.allowed_groups), profile)
    return profile, groups
