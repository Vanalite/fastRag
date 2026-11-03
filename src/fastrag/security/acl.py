"""The access filter: the one object every index must be given to search.

Indexes take ``access`` as a required keyword argument and apply it *before* ranking
(pre-filtering). Filtering after top-k would return fewer than k results and, worse,
lets scores computed over other tenants' data influence what a caller sees.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass

from fastrag.domain import Principal


@dataclass(frozen=True)
class AccessFilter:
    tenant_id: str
    tokens: frozenset[str]

    @classmethod
    def for_principal(cls, principal: Principal) -> AccessFilter:
        return cls(tenant_id=principal.tenant_id, tokens=principal.acl_tokens())

    def allows(self, tenant_id: str, acl: Iterable[str]) -> bool:
        # Deny by default: wrong tenant, or an empty / non-matching ACL, means no access.
        return tenant_id == self.tenant_id and not self.tokens.isdisjoint(acl)

    def scope_key(self) -> str:
        """Stable key for everything this filter can see; used to partition caches."""
        material = json.dumps([self.tenant_id, sorted(self.tokens)])
        return hashlib.sha256(material.encode()).hexdigest()
