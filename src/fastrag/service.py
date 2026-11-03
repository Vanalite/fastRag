"""``RAGService``: the single entry point for every operation.

The API, the CLI and the evaluation harness all call this class, and every method takes
a ``Principal``. Each call follows the same sequence:

    authorize (RBAC) -> rate limit -> do the work with an AccessFilter -> audit -> metrics

Keeping that sequence in one place is what makes the security properties checkable:
there is no second path to the data that skips it.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Literal

from fastrag.cache import AnswerCache, Cache, RedisAnswerCache, cache_key
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
    ScoredChunk,
    user_token,
)
from fastrag.embeddings import Embedder, get_embedder
from fastrag.errors import (
    ConfigurationError,
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
    Generation,
    Generator,
    looks_like_injection,
    validate_citations,
)
from fastrag.ingest import IngestService, Redactor
from fastrag.jobs import IngestJob, IngestJobRunner, MemoryJobStore, RedisJobStore
from fastrag.observability import AuditLog, JsonlAuditSink, MemoryAuditSink, MetricsRegistry
from fastrag.observability.audit import fingerprint
from fastrag.observability.logging import request_id_var
from fastrag.observability.tracing import end_span, start_span
from fastrag.ratelimit import Limiter, RateLimiter, RedisRateLimiter
from fastrag.retrieval import Reranker, Retriever, get_reranker
from fastrag.security.acl import AccessFilter
from fastrag.security.rbac import Permission, require
from fastrag.storage import BM25Index, NumpyVectorIndex, SearchIndexes, SQLiteDocumentStore
from fastrag.storage.base import DocumentStore

log = logging.getLogger(__name__)

ABSTAIN_ANSWER = "I couldn't find information about that in the documents available to you."


@dataclass(frozen=True)
class AskEvent:
    """One step of a streamed answer: a text ``delta``, or the final ``answer``."""

    type: Literal["delta", "answer"]
    text: str = ""
    answer: Answer | None = None


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
        cache: Cache | None = None,
        limiter: Limiter | None = None,
        user_limiter: Limiter | None = None,
        redis: Any | None = None,
        reranker: Reranker | None = None,
    ):
        self.settings = settings
        self.store = store
        self.indexes = indexes
        self.embedder = embedder
        self.generator = generator
        # `is None`, not `or`: caches and limiters define __len__, so an empty one is
        # falsy and `x or default()` would silently replace what the caller passed in.
        self.audit = audit if audit is not None else AuditLog()
        self.metrics = metrics if metrics is not None else MetricsRegistry()
        self.cache = (
            cache
            if cache is not None
            else AnswerCache(settings.cache_ttl_seconds, settings.cache_max_entries)
        )
        self.limiter = (
            limiter
            if limiter is not None
            else RateLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst)
        )
        self.user_limiter = (
            user_limiter
            if user_limiter is not None
            else RateLimiter(settings.user_rate_limit_per_minute, settings.user_rate_limit_burst)
        )
        self.redis = redis  # shared client, if any (the API reuses it for login throttling)
        self.jobs = IngestJobRunner(
            self._ingest_for_job,
            RedisJobStore(redis) if redis is not None else MemoryJobStore(),
            workers=settings.ingest_workers,
            max_active_per_tenant=settings.max_active_jobs_per_tenant,
            max_documents=settings.max_documents_per_job,
            max_queued_jobs=settings.max_queued_jobs,
            max_queued_chars=settings.max_queued_chars,
            pace=self._pace_job,
        )
        # Background documents draw on their own per-tenant budget, so a 500-document
        # job can't monopolize embedding capacity; workers wait rather than fail.
        self._job_limiter: Limiter = (
            RedisRateLimiter(
                redis,
                settings.job_documents_per_minute,
                settings.job_documents_burst,
                prefix="fastrag:rl:jobs:",
            )
            if redis is not None
            else RateLimiter(settings.job_documents_per_minute, settings.job_documents_burst)
        )
        self._question_redactor = Redactor()
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
            reranker=reranker if reranker is not None else get_reranker(settings.reranker),
            rerank_pool=settings.rerank_pool,
        )

    # --- construction ------------------------------------------------------

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, *, generator: Generator | None = None, **kwargs: Any
    ) -> RAGService:
        settings = settings or Settings()
        if settings.redis_url is not None and "redis" not in kwargs:
            kwargs["redis"] = connect_redis(settings.redis_url.get_secret_value())
        if kwargs.get("redis") is not None:
            client = kwargs["redis"]
            kwargs.setdefault("cache", RedisAnswerCache(client, settings.cache_ttl_seconds))
            kwargs.setdefault(
                "limiter",
                RedisRateLimiter(
                    client,
                    settings.rate_limit_per_minute,
                    settings.rate_limit_burst,
                    prefix="fastrag:rl:tenant:",
                ),
            )
            kwargs.setdefault(
                "user_limiter",
                RedisRateLimiter(
                    client,
                    settings.user_rate_limit_per_minute,
                    settings.user_rate_limit_burst,
                    prefix="fastrag:rl:user:",
                ),
            )
        embedder = get_embedder(settings.embedder)
        store: DocumentStore
        if settings.postgres_dsn is not None:
            store, indexes = _postgres_storage(settings, embedder)
        else:
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
            generator=generator if generator is not None else build_generator(settings),
            audit=AuditLog(sink),
            **kwargs,
        )

    # --- operations --------------------------------------------------------

    def ingest(self, principal: Principal, doc: DocumentIn) -> IngestResult:
        return self._ingest(principal, doc)

    def submit_ingest_job(self, principal: Principal, documents: list[DocumentIn]) -> IngestJob:
        """Queue documents for background ingestion; returns at once with the job.

        The submission is authorized, rate limited and audited; each document is then
        ingested as ``principal`` (permissions and audit apply per document).
        """
        with self._operation("submit_ingest_job", principal, Permission.INGEST) as details:
            details["documents"] = len(documents)
            job = self.jobs.submit(principal, documents)
            details["job_id"] = job.id
            return job

    def ingest_job(self, principal: Principal, job_id: str) -> IngestJob:
        """A job's progress, for its submitter or a tenant admin (404 for anyone else)."""
        with self._operation("read_ingest_job", principal, Permission.INGEST, audited=False):
            return self.jobs.get(principal, job_id)

    def _pace_job(self, tenant_id: str, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                self._job_limiter.check(tenant_id)
                return
            except RateLimited as exc:
                stop.wait(min(exc.retry_after, 5.0))

    def _ingest_for_job(self, principal: Principal, doc: DocumentIn, job_id: str) -> IngestResult:
        return self._ingest(principal, doc, rate_limited=False, job_id=job_id)

    def _ingest(
        self,
        principal: Principal,
        doc: DocumentIn,
        *,
        rate_limited: bool = True,
        job_id: str | None = None,
    ) -> IngestResult:
        with self._operation(
            "ingest", principal, Permission.INGEST, rate_limited=rate_limited
        ) as details:
            if job_id is not None:
                details["job_id"] = job_id
            access = AccessFilter.for_principal(principal)
            result, redactions = self.ingestor.ingest(
                principal.tenant_id,
                doc,
                can_modify=lambda rec: self._can_modify(principal, access, rec),
                default_editors=frozenset({user_token(principal.user_id)}),
                can_change_sharing=lambda rec: Role.ADMIN in principal.roles,
            )
            for kind, count in redactions.items():
                self.metrics.redactions.inc({"kind": kind}, count)
            if result.status != "unchanged":
                # Answers built from the old version must not be served again - and
                # their copies of its text shouldn't linger until the TTL expires.
                self.cache.invalidate_tenant(principal.tenant_id)
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
            self.cache.invalidate_tenant(principal.tenant_id)

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
            result = self._retrieve(query, AccessFilter.for_principal(principal), k, details)
            details["chunk_ids"] = [s.chunk.id for s in result.chunks]
            return result

    def ask(self, principal: Principal, question: str, k: int | None = None) -> Answer:
        events = list(self._ask(principal, question, k, stream=False))
        answer = events[-1].answer
        assert answer is not None
        return answer

    def ask_stream(
        self, principal: Principal, question: str, k: int | None = None
    ) -> Iterator[AskEvent]:
        """Like ``ask``, but yields text deltas as they are generated, then the answer.

        Deltas are the model's raw output. The final ``answer`` event carries the
        validated text and citations, and is what clients should keep. Authorization and
        rate limits run when the first event is requested, so callers can surface those
        errors before starting to stream.
        """
        return self._ask(principal, question, k, stream=True)

    def _ask(
        self, principal: Principal, question: str, k: int | None, *, stream: bool
    ) -> Iterator[AskEvent]:
        with self._operation("ask", principal, Permission.QUERY) as details:
            question, k = self._validate_question(question, k)
            details.update(self._question_details(question), k=k, generator=self.generator.name)
            if stream:
                details["streamed"] = True
            access = AccessFilter.for_principal(principal)

            generation_at_start = self.store.generation(principal.tenant_id)
            key = cache_key(access, generation_at_start, question, k)
            cached = self.cache.get(key)
            if cached is not None:
                details.update(cached=True, **self._answer_details(cached))
                if cached.abstained:
                    details["outcome"] = "abstained"
                yield AskEvent("answer", answer=replace(cached, cached=True))
                return

            retrieval = self._retrieve(question, access, k, details)
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
                outbound = self._outbound_question(question, details)
                streamer = getattr(self.generator, "stream", None)
                generation: Generation | None = None
                if stream and streamer is not None:
                    for item in streamer(outbound, sources):
                        if isinstance(item, Generation):
                            generation = item
                        else:
                            yield AskEvent("delta", text=item)
                else:
                    generation = self.generator.generate(outbound, sources)
                if generation is None:
                    raise GenerationError("the generator finished without an answer")
                elapsed = time.perf_counter() - started
                self.metrics.latency.observe(elapsed, {"stage": "generation"})
                details.setdefault("_timings", {})["generation"] = elapsed * 1000
                answer = self._grounded_answer(principal, generation, sources, details)

            details.update(self._answer_details(answer))
            # If the corpus changed while we were answering, the answer may quote a
            # document that has since been deleted or replaced: don't cache it.
            if self.store.generation(principal.tenant_id) == generation_at_start:
                self.cache.put(key, answer, principal.tenant_id)
            yield AskEvent("answer", answer=answer)

    def _grounded_answer(
        self,
        principal: Principal,
        generation: Generation,
        sources: list[ScoredChunk],
        details: dict[str, Any],
    ) -> Answer:
        """Validate citations against the sources shown, then build the answer."""
        if generation.citations is not None:
            # The generator reported its citations directly (e.g. extractive quotes,
            # whose text may itself contain "[2]"-style references).
            text = generation.text
            used = [i for i in generation.citations if 1 <= i <= len(sources)]
            invalid = [i for i in generation.citations if not 1 <= i <= len(sources)]
        else:
            text, used, invalid = validate_citations(generation.text, len(sources))
        if invalid:
            self.metrics.invalid_citations.inc({"tenant": principal.tenant_id}, len(invalid))
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
        answer = Answer(text=text, citations=citations, sources=sources, usage=generation.usage)
        if not answer.grounded:
            details["ungrounded"] = True
            self.metrics.ungrounded.inc({"tenant": principal.tenant_id})
        self._record_usage(principal, answer)
        return answer

    def audit_events(self, principal: Principal, limit: int = 100) -> list[dict[str, Any]]:
        with self._operation("read_audit", principal, Permission.READ_AUDIT):
            return self.audit.events(principal.tenant_id, limit=max(1, min(limit, 1000)))

    def close(self) -> None:
        self.jobs.shutdown()
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
        span = start_span(
            f"fastrag.{action}",
            **{"fastrag.tenant": principal.tenant_id, "fastrag.request_id": request_id_var.get()},
        )
        try:
            if rate_limited:
                # The user's own bucket first: someone hammering the API exhausts their
                # allowance without draining the budget their colleagues share.
                self.user_limiter.check(json.dumps([principal.tenant_id, principal.user_id]))
                self.limiter.check(principal.tenant_id)
            require(principal, permission)
            yield details
            outcome = details.pop("outcome", "success")
            raised = False
        except GeneratorExit:
            # A streaming client went away mid-answer.
            outcome = "cancelled"
            raise
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
            timings = details.pop("_timings", {})  # for the span, not the audit trail
            labels = {"action": action, "outcome": outcome, "tenant": principal.tenant_id}
            self.metrics.requests.inc(labels)
            self.metrics.latency.observe(time.perf_counter() - started, {"stage": action})
            try:
                if audited or outcome == "denied":
                    try:
                        self.audit.record(action, outcome, principal, **details)
                    except Exception:
                        log.exception("audit write failed for %s", action)
                        # Fail closed: an action that can't be audited must not report success.
                        # If the call already failed, keep that original error instead.
                        if not raised:
                            raise
            finally:
                # After the audit (tracing must never block it), and always.
                self._end_span(
                    span,
                    outcome,
                    **{f"fastrag.{stage}_ms": round(ms, 2) for stage, ms in timings.items()},
                    **{
                        "fastrag.shown_chunks": len(details.get("shown_chunk_ids", ()))
                        if "shown_chunk_ids" in details
                        else None,
                        "fastrag.cached": details.get("cached"),
                    },
                )

    @staticmethod
    def _end_span(span: Any, outcome: str, **attributes: Any) -> None:
        try:
            end_span(span, outcome, **attributes)
        except Exception:  # noqa: BLE001 - tracing must never break the request or its audit
            log.exception("could not end trace span")

    def _retrieve(
        self, query: str, access: AccessFilter, k: int, details: dict[str, Any]
    ) -> RetrievalResult:
        started = time.perf_counter()
        result = self.retriever.retrieve(query, access, k)
        elapsed = time.perf_counter() - started
        self.metrics.latency.observe(elapsed, {"stage": "retrieval"})
        details.setdefault("_timings", {})["retrieval"] = elapsed * 1000
        return result

    def _outbound_question(self, question: str, details: dict[str, Any]) -> str:
        """The question as sent to the generator: PII redacted unless disabled.

        Retrieval runs locally on the original question; only what leaves the system
        (the LLM request) is redacted.
        """
        if not self.settings.redact_questions:
            return question
        result = self._question_redactor.redact(question)
        if result.counts:
            details["question_redactions"] = result.counts
            for kind, count in result.counts.items():
                self.metrics.redactions.inc({"kind": kind}, count)
        return result.text

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
        """Who may replace or delete a document: tenant admins, or listed editors.

        Editors must also be able to read the document - no blind overwrites. A document
        with no editors (e.g. migrated from an older schema) is admin-only.
        """
        if Role.ADMIN in principal.roles:
            return access.tenant_id == record.tenant_id
        return access.allows(record.tenant_id, record.acl) and access.allows(
            record.tenant_id, record.editors
        )


def _postgres_storage(
    settings: Settings, embedder: Embedder, *, migrate: bool | None = None
) -> tuple[DocumentStore, SearchIndexes]:
    try:
        from fastrag.storage.postgres import PostgresDocumentStore, postgres_indexes
    except ImportError as exc:
        raise ConfigurationError(
            "FASTRAG_POSTGRES_DSN is set but the Postgres drivers are missing; "
            "run: pip install 'fastrag[postgres]'"
        ) from exc
    assert settings.postgres_dsn is not None
    store = PostgresDocumentStore(
        settings.postgres_dsn.get_secret_value(),
        dim=embedder.dim,
        pool_size=settings.postgres_pool_size,
        migrate=settings.postgres_migrate_on_start if migrate is None else migrate,
    )
    store.check_embedder(embedder.name)
    # The indexes are the store's own tables: nothing to rebuild, nothing to drift.
    return store, postgres_indexes(store, ef_search=settings.postgres_ef_search)


def migrate_database(settings: Settings) -> str:
    """Apply pending schema migrations (``fastrag db migrate``). Returns a summary."""
    embedder = get_embedder(settings.embedder)
    if settings.postgres_dsn is None:
        SQLiteDocumentStore(settings.database_path).close()  # migrates on open
        return f"SQLite schema at {settings.database_path} is current"
    store, _ = _postgres_storage(settings, embedder, migrate=True)
    try:
        return f"Postgres schema is at v{store.SCHEMA_VERSION}"
    finally:
        store.close()


def connect_redis(url: str) -> Any:
    try:
        import redis
    except ImportError as exc:
        raise ConfigurationError(
            "FASTRAG_REDIS_URL is set but the redis package is missing; "
            "run: pip install 'fastrag[redis]'"
        ) from exc
    client = redis.Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2)
    client.ping()  # fail fast at startup rather than on the first request
    return client


def build_generator(settings: Settings) -> Generator:
    if settings.generator == "claude":
        return ClaudeGenerator(model=settings.claude_model, effort=settings.claude_effort)
    return ExtractiveGenerator()
