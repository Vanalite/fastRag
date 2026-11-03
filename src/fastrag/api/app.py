"""FastAPI application.

No ``from __future__ import annotations`` here: FastAPI resolves the dependency
annotations at runtime, and the ``Caller`` alias is local to ``create_app``.

The HTTP layer stays thin: authenticate, translate JSON <-> domain objects, map errors
to status codes. Every decision about access lives in ``RAGService``.
"""

import contextlib
import json
import logging
import re
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Annotated

import anyio
from fastapi import Depends, FastAPI, Query, Request, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from starlette.exceptions import HTTPException as StarletteHTTPException

from fastrag import __version__
from fastrag.api.schemas import (
    AskResponse,
    DocumentOut,
    IngestJobOut,
    IngestJobRequest,
    IngestRequest,
    IngestResponse,
    PrincipalOut,
    QueryRequest,
    SearchResponse,
    SourceOut,
)
from fastrag.config import Settings
from fastrag.domain import DocumentIn, Principal
from fastrag.errors import (
    AuthenticationError,
    ConcurrentModification,
    ConfigurationError,
    FastRagError,
    GenerationError,
    InvalidInput,
    NotFound,
    PermissionDenied,
    RateLimited,
)
from fastrag.observability import configure_logging, request_id_var
from fastrag.ratelimit import Limiter, RateLimiter, RedisRateLimiter
from fastrag.security.auth import (
    ApiKeyStore,
    Authenticator,
    ClaimMapping,
    JwtVerifier,
    OidcVerifier,
)
from fastrag.service import AskEvent, RAGService

log = logging.getLogger(__name__)

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_API_KEY = APIKeyHeader(name="X-API-Key", auto_error=False, description="Service API key")
_BEARER = HTTPBearer(auto_error=False, description="JWT issued by your identity provider")

_STATUS = {
    AuthenticationError: 401,
    PermissionDenied: 403,
    NotFound: 404,
    ConcurrentModification: 409,
    InvalidInput: 422,
    RateLimited: 429,
    GenerationError: 502,
}


def _error(status: int, kind: str, message: str, **extra) -> JSONResponse:
    """The one error envelope every failure uses, always carrying the request id."""
    body = {"error": {"type": kind, "message": message, **extra}}
    body["request_id"] = request_id_var.get()
    return JSONResponse(body, status_code=status)


class _ThrottleAudit:
    """Turns a flood of throttled logins into one audit entry per client per interval.

    Each throttled attempt is counted; the first one, and then at most one per
    ``interval`` seconds, is written to the audit log with the number of attempts seen
    since the last entry. Attacks stay visible without the log becoming the target.
    """

    def __init__(self, interval: float = 60.0, max_clients: int = 10_000, clock=time.monotonic):
        self.interval = interval
        self.max_clients = max_clients
        self._clock = clock
        self._state: OrderedDict[str, tuple[float, int]] = OrderedDict()  # last audit, count
        self._lock = threading.Lock()

    def note(self, client: str) -> int | None:
        """Count one throttled attempt; return the count to audit now, or None."""
        now = self._clock()
        with self._lock:
            last, count = self._state.get(client, (float("-inf"), 0))
            count += 1
            if now - last >= self.interval:
                self._state[client] = (now, 0)
                result: int | None = count
            else:
                self._state[client] = (last, count)
                result = None
            self._state.move_to_end(client)
            while len(self._state) > self.max_clients:
                self._state.popitem(last=False)
            return result


def _document_in(body: IngestRequest) -> DocumentIn:
    return DocumentIn(
        source_id=body.source_id,
        text=body.text,
        acl=body.acl_tokens(),
        title=body.title,
        media_type=body.media_type,
        metadata=body.metadata,
        editors=frozenset(body.editors),
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_error(exc: Exception, request_id: str | None) -> str:
    if isinstance(exc, FastRagError) and not isinstance(exc, ConfigurationError):
        kind, message = type(exc).__name__, str(exc)
        if isinstance(exc, GenerationError):
            message = "the answer generator failed; try again later"
    else:
        log.exception("unhandled error while streaming")
        kind, message = "InternalError", "internal server error"
    return _sse("error", {"error": {"type": kind, "message": message}, "request_id": request_id})


async def _event_stream(
    first: AskEvent, events: Iterator[AskEvent], request_id: str | None
) -> AsyncIterator[str]:
    """Server-sent events for one answer, with a well-defined end in every case.

    The service generator is advanced in a worker thread (it does blocking I/O), with the
    request id restored for logs and the audit, since the request middleware has already
    returned. Three endings:

    * The answer is ready: the generator is run to completion *before* the answer is
      sent, so the audit records success even if the client leaves right after.
    * A failure mid-stream: reported in-band as an ``error`` event.
    * The client disconnects: this coroutine is cancelled and ``finally`` closes the
      generator at once, so the audit records ``cancelled`` with the right request id,
      instead of whenever the garbage collector gets to it.
    """

    def in_context(fn, *args):
        token = request_id_var.set(request_id)
        try:
            return fn(*args)
        finally:
            request_id_var.reset(token)

    def drain() -> None:
        for _ in events:
            pass

    event: AskEvent | None = first
    try:
        while event is not None:
            if event.type == "answer":
                assert event.answer is not None
                await run_in_threadpool(in_context, drain)
                yield _sse("answer", AskResponse.of(event.answer).model_dump())
                return
            yield _sse("delta", {"text": event.text})
            event = await run_in_threadpool(in_context, next, events, None)
    except Exception as exc:  # the stream has started: report in-band, then stop
        yield in_context(_stream_error, exc, request_id)
    finally:
        # Shielded: a cancelled task would otherwise abort this cleanup immediately.
        with anyio.CancelScope(shield=True):
            await run_in_threadpool(in_context, events.close)


def create_app(
    service: RAGService,
    authenticator: Authenticator,
    *,
    auth_failure_limiter: Limiter | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        service.close()

    app = FastAPI(
        title="fastRag",
        version=__version__,
        description="Enterprise RAG reference: multi-tenant, access-controlled, audited.",
        lifespan=lifespan,
    )
    app.state.service = service
    settings = service.settings
    if auth_failure_limiter is not None:
        auth_failures: Limiter = auth_failure_limiter
    elif service.redis is not None:  # one budget per address across all replicas
        auth_failures = RedisRateLimiter(
            service.redis,
            settings.auth_failures_per_minute,
            settings.auth_failure_burst,
            prefix="fastrag:rl:auth:",
        )
    else:
        auth_failures = RateLimiter(settings.auth_failures_per_minute, settings.auth_failure_burst)
    throttle_audit = _ThrottleAudit()

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
        except Exception:
            # Anything unhandled becomes a generic 500 *here*, inside the request
            # context, so it still gets the envelope, the request id and the log line.
            log.exception("unhandled error")
            response = _error(500, "InternalError", "internal server error")
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(FastRagError)
    async def domain_error(request: Request, exc: FastRagError) -> JSONResponse:
        status = next((code for cls, code in _STATUS.items() if isinstance(exc, cls)), 500)
        # Generic messages where details would help an attacker (why a credential
        # failed) or leak internals (upstream SDK errors, configuration, paths).
        if status == 401:
            message = "authentication required"
        elif status == 502:
            message = "the answer generator failed; try again later"
        elif status >= 500:
            message = "internal server error"
        else:
            message = str(exc)
        if status >= 500:
            log.error("request failed: %s", exc)
        response = _error(status, type(exc).__name__, message)
        if isinstance(exc, RateLimited):
            response.headers["Retry-After"] = str(max(1, round(exc.retry_after)))
        if isinstance(exc, AuthenticationError):
            response.headers["WWW-Authenticate"] = "Bearer"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default 422 echoes the submitted `input` - for /v1/documents that
        # is the whole document. Return only where and why validation failed.
        details = [
            {"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")}
            for e in exc.errors()
        ]
        return _error(422, "ValidationError", "request validation failed", details=details)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        response = _error(exc.status_code, "HTTPError", str(exc.detail))
        for name, value in (exc.headers or {}).items():
            response.headers[name] = value
        return response

    def current_principal(
        request: Request,
        api_key: Annotated[str | None, Security(_API_KEY)] = None,
        bearer: Annotated[HTTPAuthorizationCredentials | None, Security(_BEARER)] = None,
    ) -> Principal:
        authorization = f"Bearer {bearer.credentials}" if bearer else None
        # Brute-force protection, per client address. Failed logins spend tokens; once an
        # address has none left, it gets 429 *before* its credentials are evaluated, so
        # it can't keep guessing. Successful logins never spend tokens. The address is the
        # direct peer: behind a proxy, run uvicorn with --forwarded-allow-ips so it is the
        # real client, or every user behind the proxy shares one budget.
        client = request.client.host if request.client else "unknown"
        try:
            auth_failures.ensure_available(client)
        except RateLimited:
            service.metrics.requests.inc(
                {"action": "authenticate", "outcome": "throttled", "tenant": "-"}
            )
            attempts = throttle_audit.note(client)
            if attempts is not None:
                service.audit.record(
                    "authenticate", "throttled", None, client=client, attempts=attempts
                )
            raise
        try:
            return authenticator.authenticate(authorization, api_key)
        except AuthenticationError as exc:
            with contextlib.suppress(RateLimited):  # a concurrent failure took the last token
                auth_failures.check(client)
            service.audit.record(
                "authenticate",
                "unauthenticated",
                None,
                path=request.url.path,
                client=client,
                reason=str(exc),
            )
            service.metrics.requests.inc(
                {"action": "authenticate", "outcome": "unauthenticated", "tenant": "-"}
            )
            raise

    Caller = Annotated[Principal, Depends(current_principal)]

    # --- operational endpoints (no auth: probes and scrapers) ---------------

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"])
    def readyz() -> Response:
        ready = service.ready()
        return JSONResponse({"ready": ready}, status_code=200 if ready else 503)

    if service.settings.expose_metrics:

        @app.get("/metrics", tags=["ops"], response_class=PlainTextResponse)
        def metrics() -> str:
            return service.metrics.render()

    # --- API ------------------------------------------------------------------

    @app.get("/v1/me", tags=["identity"])
    def me(principal: Caller) -> PrincipalOut:
        return PrincipalOut(
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            groups=sorted(principal.groups),
            roles=sorted(r.value for r in principal.roles),
        )

    @app.post("/v1/documents", tags=["documents"], status_code=201)
    def ingest(body: IngestRequest, principal: Caller) -> IngestResponse:
        result = service.ingest(principal, _document_in(body))
        return IngestResponse(**vars(result))

    @app.post("/v1/ingest-jobs", tags=["documents"], status_code=202)
    def submit_ingest_job(
        body: IngestJobRequest, principal: Caller, response: Response
    ) -> IngestJobOut:
        """Queue up to 500 documents for background ingestion. Poll the returned job."""
        job = service.submit_ingest_job(principal, [_document_in(d) for d in body.documents])
        response.headers["Location"] = f"/v1/ingest-jobs/{job.id}"
        return IngestJobOut.of(job)

    @app.get("/v1/ingest-jobs/{job_id}", tags=["documents"])
    def get_ingest_job(job_id: str, principal: Caller) -> IngestJobOut:
        return IngestJobOut.of(service.ingest_job(principal, job_id))

    @app.get("/v1/documents", tags=["documents"])
    def list_documents(
        principal: Caller,
        response: Response,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> list[DocumentOut]:
        docs = service.list_documents(principal)
        response.headers["X-Total-Count"] = str(len(docs))
        return [DocumentOut.of(d) for d in docs[offset : offset + limit]]

    @app.delete("/v1/documents/{doc_id}", tags=["documents"], status_code=204)
    def delete_document(doc_id: str, principal: Caller) -> Response:
        service.delete(principal, doc_id)
        return Response(status_code=204)

    @app.post("/v1/search", tags=["query"])
    def search(body: QueryRequest, principal: Caller) -> SearchResponse:
        result = service.search(principal, body.question, body.k)
        return SearchResponse(
            confident=result.confident, results=[SourceOut.of(s) for s in result.chunks]
        )

    @app.post("/v1/ask", tags=["query"])
    def ask(body: QueryRequest, principal: Caller) -> AskResponse:
        return AskResponse.of(service.ask(principal, body.question, body.k))

    @app.post(
        "/v1/ask/stream",
        tags=["query"],
        response_class=StreamingResponse,
        responses={200: {"content": {"text/event-stream": {}}}},
    )
    def ask_stream(body: QueryRequest, principal: Caller) -> StreamingResponse:
        """Server-sent events: `delta` events with text as it is generated, then one
        `answer` event (same shape as `/v1/ask`) carrying the validated answer and
        citations - keep that one. Failures after streaming starts arrive as `error`."""
        events = service.ask_stream(principal, body.question, body.k)
        # Run up to the first event now: authorization, rate limits, validation and
        # retrieval errors then become ordinary HTTP errors instead of a 200 stream.
        first = next(events)
        return StreamingResponse(
            _event_stream(first, events, request_id_var.get()),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},  # don't let proxies buffer the stream
        )

    @app.get("/v1/audit", tags=["governance"])
    def audit(
        principal: Caller, limit: Annotated[int, Query(ge=1, le=1000)] = 100
    ) -> dict[str, list[dict]]:
        return {"events": service.audit_events(principal, limit)}

    return app


def create_app_from_settings(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level, settings.log_json)
    service = RAGService.from_settings(settings)
    return create_app(service, build_authenticator(settings))


def build_authenticator(settings: Settings) -> Authenticator:
    claims = ClaimMapping(
        tenant=settings.token_tenant_claim,
        groups=settings.token_groups_claim,
        roles=settings.token_roles_claim,
    )
    jwt_verifier = (
        JwtVerifier(
            settings.jwt_secret.get_secret_value(),
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
            claims=claims,
        )
        if settings.jwt_secret
        else None
    )
    oidc_verifier = (
        OidcVerifier(
            settings.oidc_jwks_url,
            issuer=settings.jwt_issuer or "",
            audience=settings.jwt_audience or "",
            claims=claims,
        )
        if settings.oidc_jwks_url
        else None
    )
    api_keys = ApiKeyStore(settings.api_keys_file) if settings.api_keys_file else None
    return Authenticator(api_keys=api_keys, jwt_verifier=jwt_verifier, oidc_verifier=oidc_verifier)
