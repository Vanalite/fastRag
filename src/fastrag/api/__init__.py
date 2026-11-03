"""HTTP API. Run with ``fastrag serve`` or ``uvicorn fastrag.api:app``."""

from typing import Any

from fastrag.api.app import create_app, create_app_from_settings

__all__ = ["create_app", "create_app_from_settings"]


def __getattr__(name: str) -> Any:
    # `uvicorn fastrag.api:app` builds the app lazily from FASTRAG_* settings, so that
    # importing this package (e.g. in tests) has no side effects.
    if name == "app":
        return create_app_from_settings()
    raise AttributeError(name)
