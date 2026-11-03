"""Shared fixtures for the security acceptance suite (no network, no sleeps)."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from fastrag.config import Settings
from fastrag.domain import (
    EVERYONE,
    DocumentIn,
    Principal,
    Role,
    ScoredChunk,
    Usage,
)
from fastrag.generation import Generation
from fastrag.service import RAGService

CANARY_A = "CANARY-TENANT-A-91bc"
CANARY_B = "CANARY-TENANT-B-7f3a"


class ScriptedGenerator:
    """Fake LLM: records everything it is shown and replies with scripted text."""

    name = "scripted"

    def __init__(self, reply: str | Callable[[str, list[ScoredChunk]], str] = "Answer [1]."):
        self.reply = reply
        self.calls: list[tuple[str, list[ScoredChunk]]] = []

    def generate(self, question: str, sources: list[ScoredChunk]) -> Generation:
        self.calls.append((question, list(sources)))
        text = self.reply(question, sources) if callable(self.reply) else self.reply
        return Generation(text=text, usage=Usage(input_tokens=1, output_tokens=1))

    @property
    def shown_texts(self) -> list[str]:
        return [s.chunk.text for _, srcs in self.calls for s in srcs]

    @property
    def shown_chunk_ids(self) -> set[str]:
        return {s.chunk.id for _, srcs in self.calls for s in srcs}


def make_principal(
    tenant: str = "tenant-a",
    user: str = "alice",
    groups: tuple[str, ...] = (),
    roles: tuple[Role, ...] = (Role.READER,),
) -> Principal:
    return Principal(tenant, user, frozenset(groups), frozenset(roles))


def doc(source_id: str, text: str, acl=(EVERYONE,), title: str = "") -> DocumentIn:
    return DocumentIn(source_id=source_id, text=text, acl=frozenset(acl), title=title)


def audit_dump(service: RAGService) -> str:
    return json.dumps(service.audit.sink.read(), sort_keys=True, default=str)


@pytest.fixture
def generator() -> ScriptedGenerator:
    return ScriptedGenerator()


@pytest.fixture
def make_service(generator):
    def _make(**overrides) -> RAGService:
        overrides.setdefault("rate_limit_per_minute", 0)  # disabled unless a test opts in
        return RAGService.from_settings(Settings.for_testing(**overrides), generator=generator)

    return _make


@pytest.fixture
def service(make_service) -> RAGService:
    return make_service()


@pytest.fixture
def admin_a() -> Principal:
    return make_principal("tenant-a", "root-a", roles=(Role.ADMIN,))


@pytest.fixture
def admin_b() -> Principal:
    return make_principal("tenant-b", "root-b", roles=(Role.ADMIN,))


@pytest.fixture
def alice() -> Principal:
    return make_principal("tenant-a", "alice", groups=("finance",))


@pytest.fixture
def bob() -> Principal:
    return make_principal("tenant-a", "bob")
