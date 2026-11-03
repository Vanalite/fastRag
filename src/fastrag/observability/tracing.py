"""OpenTelemetry spans, when ``opentelemetry-api`` is installed (``fastrag[otel]``).

Without it, every function here is a no-op, so tracing costs nothing in the default
install. To export traces, install the SDK and an exporter in your deployment and
configure them the usual way (e.g. ``opentelemetry-instrument`` and ``OTEL_*``
environment variables); with FastAPI instrumentation, operation spans nest under the
HTTP request span.

Each service operation gets one span, started and ended explicitly rather than made
"current": a streamed answer is produced across several worker threads, and current-span
context can't be carried safely between them. Stage timings (retrieval, generation) are
recorded as attributes instead of child spans.

Attributes are deliberately coarse: tenant, action, outcome, counts, timings and the
request id. Never the question, document text or user id - traces often flow to
third-party backends with weaker access control than the audit log.
"""

from __future__ import annotations

from typing import Any

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
except ImportError:  # pragma: no cover - exercised when the extra isn't installed
    trace = None  # type: ignore[assignment]

_ERROR_OUTCOMES = frozenset({"error", "generation_error"})


def start_span(name: str, **attributes: Any) -> Any:
    """Start a span (parented on the caller's current span, if any) or return None."""
    if trace is None:
        return None
    clean = {k: v for k, v in attributes.items() if v is not None}
    return trace.get_tracer("fastrag").start_span(name, attributes=clean)


def end_span(span: Any, outcome: str, **attributes: Any) -> None:
    if span is None:
        return
    span.set_attributes({"fastrag.outcome": outcome})
    span.set_attributes({k: v for k, v in attributes.items() if v is not None})
    if outcome in _ERROR_OUTCOMES:
        span.set_status(Status(StatusCode.ERROR, outcome))
    span.end()
