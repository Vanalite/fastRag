"""fastRag: a reference implementation of an enterprise-grade RAG system."""

__version__ = "0.2.0"

from fastrag.config import Settings  # noqa: E402
from fastrag.domain import (  # noqa: E402
    EVERYONE,
    Answer,
    DocumentIn,
    Principal,
    Role,
    group_token,
    user_token,
)
from fastrag.service import RAGService  # noqa: E402

__all__ = [
    "EVERYONE",
    "Answer",
    "DocumentIn",
    "Principal",
    "RAGService",
    "Role",
    "Settings",
    "__version__",
    "group_token",
    "user_token",
]
