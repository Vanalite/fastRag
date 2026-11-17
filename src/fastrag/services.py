"""Process-wide dependencies, built once at startup and swappable in tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from fastrag.config import Settings, get_settings
from fastrag.ingest.ocr import VlmOcr, build_ocr
from fastrag.providers.embeddings import Embedder, build_embedder
from fastrag.providers.llm import ChatLLM, build_llms, default_llm
from fastrag.providers.reranker import Reranker, build_reranker
from fastrag.providers.tokens import get_counter
from fastrag.storage import Storage, build_storage
from fastrag.text.values import Gazetteer, default_gazetteer


@dataclass
class Services:
    settings: Settings
    embedder: Embedder
    reranker: Reranker | None
    llm: ChatLLM | None
    ocr: VlmOcr | None
    storage: Storage
    gazetteer: Gazetteer
    count_tokens: Callable[[str], int]
    # Every named LLM provider (D25); `llm` is the default one.
    llms: dict[str, ChatLLM] = field(default_factory=dict)

    async def aclose(self) -> None:
        await self.embedder.aclose()
        if self.reranker:
            await self.reranker.aclose()
        for llm in {id(x): x for x in [*self.llms.values(), *([self.llm] if self.llm else [])]}.values():
            await llm.aclose()


def build_services(settings: Settings | None = None) -> Services:
    s = settings or get_settings()
    llms = build_llms(s)
    return Services(
        settings=s,
        embedder=build_embedder(s),
        reranker=build_reranker(s),
        llm=default_llm(llms),
        ocr=build_ocr(s),
        storage=build_storage(s),
        gazetteer=default_gazetteer(),
        count_tokens=get_counter(s.tokenizer),
        llms=llms,
    )
