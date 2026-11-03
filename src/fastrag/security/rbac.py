"""Role-based permissions: what a principal may *do* (ACLs decide what it may *see*)."""

from __future__ import annotations

from enum import StrEnum

from fastrag.domain import Principal, Role
from fastrag.errors import PermissionDenied


class Permission(StrEnum):
    QUERY = "query"
    INGEST = "ingest"
    DELETE = "delete"
    READ_AUDIT = "read_audit"


ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.READER: frozenset({Permission.QUERY}),
    Role.INGESTER: frozenset({Permission.QUERY, Permission.INGEST, Permission.DELETE}),
    Role.ADMIN: frozenset(Permission),
}


def permissions_for(principal: Principal) -> frozenset[Permission]:
    granted: set[Permission] = set()
    for role in principal.roles:
        granted |= ROLE_PERMISSIONS.get(role, frozenset())
    return frozenset(granted)


def require(principal: Principal, permission: Permission) -> None:
    if permission not in permissions_for(principal):
        raise PermissionDenied(f"'{principal.user_id}' lacks the '{permission}' permission")
