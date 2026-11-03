"""``RAGService``: the single entry point for every operation.

The API, the CLI and the evaluation harness all call this class, and every method takes
a ``Principal``. Each call follows the same sequence:

    authorize (RBAC) -> rate limit -> do the work with an AccessFilter -> audit -> metrics

Keeping that sequence in one place is what makes the security properties checkable:
there is no second path to the data that skips it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from fastrag.cache import AnswerCache
from fastrag.config import Settings
from fastrag.domain import (
    Answer,
    Citation,
    DocumentIn,
    DocumentRecord,
    IngestResult,
    Principal,
    RetrievalResult,
    Role,
)
from fastrag.embeddings import Embedder, get_embedder
from fastrag.errors import (
    FastRagError,
    GenerationError,
    InvalidInput,
    NotFound,
    PermissionDenied,
    RateLimited,
)
from fastrag.generation import (
    ClaudeGenerator,
    ExtractiveGenerator,
    Generator,
    looks_like_injection,
    validate_citations,
)
from fastrag.ingest import IngestService, Redactor
from fastrag.observability import AuditLog, JsonlAuditSink, MemoryAuditSink, MetricsRegistry
from fastrag.observability.audit import fingerprint
from fastrag.ratelimit import RateLimiter
from fastrag.retrieval import Retriever
from fastrag.security.acl import AccessFilter
from fastrag.security.rbac import Permission, require
from fastrag.storage import BM25Index, NumpyVectorIndex, SearchIndexes, SQLiteDocumentStore
from fastrag.storage.base import DocumentStore

log = logging.getLogger(__name__)

ABSTAIN_ANSWER = "I couldn't find information about that in the documents available to you."


class RAGService:
    def __init__(
        self,
        *,
        settings: Settings,
        store: DocumentStore,
        indexes: SearchIndexes,
        embedder: Embedder,
        generator: Generator,
        audit: AuditLog | None = None,
        metrics: MetricsRegistry | None = None,
        cache: AnswerCache | None = None,
        limiter: RateLimiter | None = None,
    ):
        self.settings = settings
        self.store = store
        self.indexes = indexes
        self.embedder = embedder
        self.generator = generator
        self.audit = audit or AuditLog()
        self.metrics = metrics or MetricsRegistry()
        self.cache = cache or AnswerCache(settings.cache_ttl_seconds, settings.cache_max_entries)
        self.limiter = limiter or RateLimiter(
            settings.rate_limit_per_minute, settings.rate_limit_burst
        )
        self.ingestor = IngestService(
            store,
            indexes,
            embedder,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            redactor=Redactor() if settings.redact_pii else None,
            max_document_chars=settings.max_document_chars,
        )
        self.retriever = Retriever(
            embedder,
            indexes,
            candidate_pool=settings.candidate_pool,
            rrf_k=settings.rrf_k,
            min_dense_score=settings.min_dense_score,
            min_keyword_coverage=settings.min_keyword_coverage,
        )

    # --- construction ------------------------------------------------------

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, *, generator: Generator | None = None, **kwargs: Any
    ) -> RAGService:
        settings = settings or Settings()
        embedder = get_embedder(settings.embedder)
        store = SQLiteDocumentStore(settings.database_path)
        store.check_embedder(embedder.name)
        indexes = SearchIndexes(NumpyVectorIndex(embedder.dim), BM25Index())
        loaded = indexes.rebuild_from(store, embedder.dim)
        log.info("indexes rebuilt from store", extra={"chunks": loaded})
        sink = (
            JsonlAuditSink(settings.audit_log_path)
            if settings.audit_log_path
            else MemoryAuditSink()
        )
        return cls(
            settings=settings,
            store=store,
            indexes=indexes,
            embedder=embedder,
            generator=generator or build_generator(settings),
            audit=AuditLog(sink),
            **kwargs,
        )

    # --- operations --------------------------------------------------------

    def ingest(self, principal: Principal, doc: DocumentIn) -> IngestResult:
        with self._operation("ingest", principal, Permission.INGEST) as details:
            access = AccessFilter.for_principal(principal)
            result, redactions = self.ingestor.ingest(
                principal.tenant_id,
                doc,
                can_modify=lambda rec: self._can_modify(principal, access, rec),
            )
            for kind, count in redactions.items():
                self.metrics.redactions.inc({"kind": kind}, count)
            details.update(
                doc_id=result.doc_id,
                version=result.version,
                status=result.status,
                chunks=result.chunks,
                acl_entries=len(doc.acl),
                redactions=redactions,
            )
            return result

    def delete(self, principal: Principal, doc_id: str) -> None:
        with self._operation("delete", principal, Permission.DELETE) as details:
            details["doc_id"] = doc_id
            access = AccessFilter.for_principal(principal)
            deleted = self.ingestor.delete(
                principal.tenant_id,
                doc_id,
                can_modify=lambda rec: self._can_modify(principal, access, rec),
            )
            if not deleted:
                # Same response whether it doesn't exist or the caller can't see it.
                raise NotFound(f"document {doc_id!r} not found")

    def list_documents(self, principal: Principal) -> list[DocumentRecord]:
        with self._operation("list_documents", principal, Permission.QUERY) as details:
            # Admins manage the whole tenant's corpus, so they see every document's
            # metadata. Content (search, ask) stays governed by ACLs for everyone.
            access = AccessFilter.for_principal(principal)
            is_admin = Role.ADMIN in principal.roles
            docs = [
                d for d in self.store.list_documents(principal.tenant_id)
                if is_admin or access.allows(d.tenant_id, d.acl)
            ]  # fmt: skip
            details["count"] = len(docs)
            return docs

    def search(self, principal: Principal, query: str, k: int | None = None) -> RetrievalResult:
        with self._operation("search", principal, Permission.QUERY) as details:
            query, k = self._validate_question(query, k)
            details.update(self._question_details(query), k=k)
            result = self._retrieve(query, AccessFilter.for_principal(principal), k)
            details["chunk_ids"] = [s.chunk.id for s in result.chunks]
            return result

    def ask(self, principal: Principal, question: str, k: int | None = None) -> Answer:
        with self._operation("ask", principal, Permission.QUERY) as details:
            question, k = self._validate_question(question, k)
            details.update(self._question_details(question), k=k, generator=self.generator.name)
            access = AccessFilter.for_principal(principal)

            cache_key = AnswerCache.key(
                access, self.store.generation(principal.tenant_id), question, k
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                details.update(cached=True, **self._answer_details(cached))
                if cached.abstained:
                    details["outcome"] = "abstained"
                return replace(cached, cached=True)

            retrieval = self._retrieve(question, access, k)
            # Only chunks that are real evidence go to the LLM; nearest-but-unrelated
            # neighbours would just invite it to pad the answer.
            sources = [s for s in retrieval.chunks if self.retriever.supports(s)]
            flagged = [
                s.chunk.id
                for s in sources
                if looks_like_injection(s.chunk.text) or looks_like_injection(s.chunk.title)
            ]
            if flagged:
                self.metrics.injection_flags.inc({"tenant": principal.tenant_id}, len(flagged))
                details["injection_flagged_chunk_ids"] = flagged
                if self.settings.drop_suspicious_chunks:
                    sources = [s for s in sources if s.chunk.id not in flagged]

            if not sources:
                # Abstain without calling the LLM: no evidence, no answer.
                answer = Answer(text=ABSTAIN_ANSWER, citations=[], sources=[], abstained=True)
                details["outcome"] = "abstained"
            else:
                started = time.perf_counter()
                generation = self.generator.generate(question, sources)
                self.metrics.latency.observe(time.perf_counter() - started, {"stage": "generation"})
                if generation.citations is not None:
                    # The generator reported its citations directly (e.g. extractive quotes,
                    # whose text may itself contain "[2]"-style references).
                    text = generation.text
                    used = [i for i in generation.citations if 1 <= i <= len(sources)]
                    invalid = [i for i in generation.citations if not 1 <= i <= len(sources)]
                else:
                    text, used, invalid = validate_citations(generation.text, len(sources))
                if invalid:
                    self.metrics.invalid_citations.inc(
                        {"tenant": principal.tenant_id}, len(invalid)
                    )
                    details["invalid_citations"] = invalid
                citations = [
                    Citation(
                        index=i,
                        chunk_id=sources[i - 1].chunk.id,
                        doc_id=sources[i - 1].chunk.doc_id,
                        source_id=sources[i - 1].chunk.source_id,
                        title=sources[i - 1].chunk.title,
                    )
                    for i in used
                ]
                answer = Answer(
                    text=text, citations=citations, sources=sources, usage=generation.usage
                )
                self._record_usage(principal, answer)

            details.update(self._answer_details(answer))
            self.cache.put(cache_key, answer)
            return answer

    def audit_events(self, principal: Principal, limit: int = 100) -> list[dict[str, Any]]:
        with self._operation("read_audit", principal, Permission.READ_AUDIT):
            return self.audit.events(principal.tenant_id, limit=max(1, min(limit, 1000)))

    def close(self) -> None:
        close = getattr(self.store, "close", None)
        if close is not None:
            close()

    def ready(self) -> bool:
        try:
            self.store.ping()
            return True
        except Exception:  # noqa: BLE001 - readiness must report, not raise
            log.exception("readiness check failed")
            return False

    # --- internals ---------------------------------------------------------

    @contextmanager
    def _operation(
        self,
        action: str,
        principal: Principal,
        permission: Permission,
        *,
        rate_limited: bool = True,
        audited: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """Rate-limit, authorize, then audit and count the outcome however the call ends.

        Rate limiting comes first so that a caller hammering an endpoint it isn't allowed
        to use is throttled too, instead of generating unlimited audit writes.
        """
        details: dict[str, Any] = {}
        outcome = "success"
        raised = True
        started = time.perf_counter()
        try:
            if rate_limited:
                self.limiter.check(principal.tenant_id)
            require(principal, permission)
            yield details
            outcome = details.pop("outcome", "success")
            raised = False
        except PermissionDenied:
            outcome = "denied"
            raise
        except RateLimited:
            outcome = "rate_limited"
            raise
        except NotFound:
            outcome = "not_found"
            raise
        except InvalidInput:
            outcome = "invalid"
            raise
        except GenerationError:
            outcome = "generation_error"
            raise
        except FastRagError:
            outcome = "error"
            raise
        except Exception:
            outcome = "error"
            log.exception("unexpected error during %s", action)
            raise
        finally:
            labels = {"action": action, "outcome": outcome, "tenant": principal.tenant_id}
            self.metrics.requests.inc(labels)
            self.metrics.latency.observe(time.perf_counter() - started, {"stage": action})
            if audited or outcome == "denied":
                try:
                    self.audit.record(action, outcome, principal, **details)
                except Exception:
                    log.exception("audit write failed for %s", action)
                    # Fail closed: an action that can't be audited must not report success.
                    # If the call already failed, keep that original error instead.
                    if not raised:
                        raise

    def _retrieve(self, query: str, access: AccessFilter, k: int) -> RetrievalResult:
        started = time.perf_counter()
        result = self.retriever.retrieve(query, access, k)
        self.metrics.latency.observe(time.perf_counter() - started, {"stage": "retrieval"})
        return result

    def _validate_question(self, question: str, k: int | None) -> tuple[str, int]:
        question = question.strip()
        if not question:
            raise InvalidInput("question must not be empty")
        if len(question) > self.settings.max_question_chars:
            raise InvalidInput(f"question exceeds {self.settings.max_question_chars} characters")
        k = self.settings.top_k if k is None else k
        if not 1 <= k <= 20:
            raise InvalidInput("k must be between 1 and 20")
        return question, k

    def _question_details(self, question: str) -> dict[str, Any]:
        details: dict[str, Any] = {
            "question_sha256": fingerprint(question),
            "question_chars": len(question),
        }
        if self.settings.audit_store_query_text:
            details["question"] = question
        return details

    @staticmethod
    def _answer_details(answer: Answer) -> dict[str, Any]:
        return {
            "abstained": answer.abstained,
            "shown_chunk_ids": [s.chunk.id for s in answer.sources],
            "shown_doc_ids": sorted({s.chunk.doc_id for s in answer.sources}),
            "cited_doc_ids": sorted({c.doc_id for c in answer.citations}),
            "usage": vars(answer.usage),
        }

    def _record_usage(self, principal: Principal, answer: Answer) -> None:
        for kind, value in vars(answer.usage).items():
            if value:
                self.metrics.tokens.inc({"tenant": principal.tenant_id, "kind": kind}, value)

    @staticmethod
    def _can_modify(principal: Principal, access: AccessFilter, record: DocumentRecord) -> bool:
        """Replacing or deleting a document requires being able to read it (or admin)."""
        return Role.ADMIN in principal.roles or access.allows(record.tenant_id, record.acl)


def build_generator(settings: Settings) -> Generator:
    if settings.generator == "claude":
        return ClaudeGenerator(model=settings.claude_model, effort=settings.claude_effort)
    return ExtractiveGenerator()
