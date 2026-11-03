"""FastAPI application.

No ``from __future__ import annotations`` here: FastAPI resolves the dependency
annotations at runtime, and the ``Caller`` alias is local to ``create_app``.

The HTTP layer stays thin: authenticate, translate JSON <-> domain objects, map errors
to status codes. Every decision about access lives in ``RAGService``.
"""

import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Query, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from starlette.exceptions import HTTPException as StarletteHTTPException

from fastrag import __version__
from fastrag.api.schemas import (
    AskResponse,
    DocumentOut,
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
    FastRagError,
    GenerationError,
    InvalidInput,
    NotFound,
    PermissionDenied,
    RateLimited,
)
from fastrag.observability import configure_logging, request_id_var
from fastrag.security.auth import ApiKeyStore, Authenticator, JwtVerifier
from fastrag.service import RAGService

log = logging.getLogger(__name__)

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_API_KEY = APIKeyHeader(name="X-API-Key", auto_error=False, description="Service API key")
_BEARER = HTTPBearer(auto_error=False, description="JWT issued by your identity provider")

_STATUS = {
    AuthenticationError: 401,
    PermissionDenied: 403,
    NotFound: 404,
    InvalidInput: 422,
    RateLimited: 429,
    GenerationError: 502,
}


def _error(status: int, kind: str, message: str, **extra) -> JSONResponse:
    """The one error envelope every failure uses, always carrying the request id."""
    body = {"error": {"type": kind, "message": message, **extra}}
    body["request_id"] = request_id_var.get()
    return JSONResponse(body, status_code=status)


def create_app(service: RAGService, authenticator: Authenticator) -> FastAPI:
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
        try:
            return authenticator.authenticate(authorization, api_key)
        except AuthenticationError as exc:
            service.audit.record(
                "authenticate", "unauthenticated", None, path=request.url.path, reason=str(exc)
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
        result = service.ingest(
            principal,
            DocumentIn(
                source_id=body.source_id,
                text=body.text,
                acl=body.acl_tokens(),
                title=body.title,
                media_type=body.media_type,
                metadata=body.metadata,
            ),
        )
        return IngestResponse(**vars(result))

    @app.get("/v1/documents", tags=["documents"])
    def list_documents(principal: Caller) -> list[DocumentOut]:
        return [DocumentOut.of(d) for d in service.list_documents(principal)]

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
    jwt_verifier = (
        JwtVerifier(
            settings.jwt_secret.get_secret_value(),
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
        )
        if settings.jwt_secret
        else None
    )
    api_keys = ApiKeyStore(settings.api_keys_file) if settings.api_keys_file else None
    return create_app(service, Authenticator(api_keys=api_keys, jwt_verifier=jwt_verifier))
