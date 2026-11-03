"""Logs, metrics and the audit trail."""

from fastrag.observability.audit import AuditLog, JsonlAuditSink, MemoryAuditSink, verify_chain
from fastrag.observability.logging import configure_logging, request_id_var
from fastrag.observability.metrics import MetricsRegistry

__all__ = [
    "AuditLog",
    "JsonlAuditSink",
    "MemoryAuditSink",
    "MetricsRegistry",
    "configure_logging",
    "request_id_var",
    "verify_chain",
]
