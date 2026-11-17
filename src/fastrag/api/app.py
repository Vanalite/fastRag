"""FastAPI application factory."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text
from starlette.exceptions import HTTPException as StarletteHTTPException

from fastrag import __version__, metrics
from fastrag.api import admin_routes, retrieval_routes
from fastrag.config import get_settings
from fastrag.db.engine import dispose_engine, get_sessionmaker
from fastrag.log import configure_logging, log_event, trace_id_var
from fastrag.services import Services, build_services

logger = logging.getLogger("fastrag.api")
_TRACE_OK = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


def create_app(services: Services | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level, settings.log_json)
        app.state.services = services or build_services(settings)
        if settings.env == "prod" and settings.admin_token.get_secret_value() == "change-me-admin":
            raise RuntimeError("set FASTRAG_ADMIN_TOKEN before running in production")
        if app.state.services.embedder.model == "hash" and settings.env == "prod":
            raise RuntimeError("the hash embedder is for tests only; configure FASTRAG_EMBEDDING_PROVIDER")
        log_event(
            logger,
            "api started",
            version=__version__,
            embedder=app.state.services.embedder.model,
            reranker=getattr(app.state.services.reranker, "name", None),
        )
        yield
        if services is None:
            await app.state.services.aclose()
        await dispose_engine()

    app = FastAPI(
        title="fastRag",
        version=__version__,
        description="Retrieval service for AI agents: hybrid document search, typed listings, evidence signals.",
        lifespan=lifespan,
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"]
        )

    @app.middleware("http")
    async def trace_and_log(request: Request, call_next: Any) -> Any:
        incoming = request.headers.get("x-trace-id", "")
        tid = incoming if _TRACE_OK.match(incoming) else uuid.uuid4().hex
        request.state.trace_id = tid
        token = trace_id_var.set(tid)
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Trace-Id"] = tid
            return response
        finally:
            elapsed = time.perf_counter() - t0
            route = getattr(request.scope.get("route"), "path", None) or "unmatched"
            if request.url.path not in ("/healthz", "/readyz", "/metrics"):
                log_event(
                    logger,
                    "request",
                    method=request.method,
                    path=request.url.path,
                    status=status,
                    ms=round(elapsed * 1000, 1),
                )
                metrics.HTTP_REQUESTS.labels(request.method, route, str(status)).inc()
                metrics.HTTP_LATENCY.labels(route).observe(elapsed)
            trace_id_var.reset(token)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail: dict[str, Any] = exc.detail if isinstance(exc.detail, dict) else {"error": exc.detail}
        detail.setdefault("trace_id", getattr(request.state, "trace_id", None))
        return JSONResponse(detail, status_code=exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
        return JSONResponse(
            {
                "error": "invalid_request",
                "details": errors,
                "trace_id": getattr(request.state, "trace_id", None),
            },
            status_code=422,
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error")
        return JSONResponse(
            {"error": "internal_error", "trace_id": getattr(request.state, "trace_id", None)}, status_code=500
        )

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"])
    async def readyz() -> JSONResponse:
        checks: dict[str, str] = {}
        try:
            async with get_sessionmaker()() as s:
                await s.execute(text("SELECT 1"))
                version = (await s.execute(text("SELECT version_num FROM alembic_version"))).scalar()
            checks["database"] = "ok"
            checks["schema"] = str(version)
        except Exception as exc:
            checks["database"] = f"error: {type(exc).__name__}"
        ok = checks.get("database") == "ok"
        return JSONResponse({"status": "ok" if ok else "unavailable", "checks": checks}, status_code=200 if ok else 503)

    admin_page = (Path(__file__).parent / "static" / "admin.html").read_text(encoding="utf-8")

    @app.get("/admin", include_in_schema=False)
    async def admin_ui() -> HTMLResponse:
        """Admin page for non-technical staff. It only calls the admin API with
        the token the user types in; the page itself holds no data."""
        return HTMLResponse(
            admin_page,
            headers={
                "Content-Security-Policy": "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                "frame-ancestors 'none'",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
            },
        )

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    async def metrics_endpoint() -> Response:
        try:
            async with get_sessionmaker()() as s:
                row = (
                    await s.execute(
                        text(
                            "SELECT (SELECT count(*) FROM jobs WHERE status = 'queued') AS queued, "
                            "(SELECT count(*) FROM jobs WHERE status = 'failed') AS failed, "
                            "(SELECT count(*) FROM documents WHERE status IN ('needs_review', 'failed')) AS review"
                        )
                    )
                ).one()
            metrics.QUEUE_DEPTH.set(row.queued)
            metrics.JOBS_FAILED.set(row.failed)
            metrics.DOCS_NEED_REVIEW.set(row.review)
        except Exception:
            logger.warning("metrics: database gauges unavailable")
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.include_router(retrieval_routes.router)
    app.include_router(admin_routes.router)
    app.include_router(admin_routes.hooks)
    return app
