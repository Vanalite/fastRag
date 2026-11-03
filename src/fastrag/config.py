"""Settings, from ``FASTRAG_*`` environment variables or a ``.env`` file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FASTRAG_", env_file=".env", extra="ignore")

    # Storage
    database_path: str = ".fastrag/fastrag.db"  # ":memory:" for an ephemeral store
    embedder: str = "hashing"

    # Ingestion
    chunk_size: int = Field(800, ge=100, le=8000)
    chunk_overlap: int = Field(100, ge=0)
    redact_pii: bool = True
    max_document_chars: int = Field(2_000_000, ge=1)

    # Retrieval
    top_k: int = Field(4, ge=1, le=20)
    candidate_pool: int = Field(20, ge=1, le=200)
    rrf_k: int = Field(60, ge=1)
    min_dense_score: float | None = Field(None, ge=-1, le=1)  # None: embedder default
    min_keyword_coverage: float = Field(0.3, ge=0, le=1)

    # Generation
    generator: Literal["extractive", "claude"] = "extractive"
    claude_model: str = "claude-opus-5-5"
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_question_chars: int = Field(2000, ge=1)
    drop_suspicious_chunks: bool = False

    # Security
    api_keys_file: Path | None = Path(".fastrag/api_keys.json")
    jwt_secret: SecretStr | None = None
    jwt_issuer: str | None = None
    jwt_audience: str | None = None

    # Operations
    audit_log_path: Path | None = Path(".fastrag/audit.jsonl")
    audit_store_query_text: bool = False
    cache_ttl_seconds: float = Field(300, ge=0)
    cache_max_entries: int = Field(1000, ge=0)
    rate_limit_per_minute: float = Field(120, ge=0)
    rate_limit_burst: int = Field(30, ge=0)
    log_level: str = "INFO"
    log_json: bool = True
    # /metrics is unauthenticated and labels series by tenant: enable it only where the
    # port is reachable from your monitoring network alone.
    expose_metrics: bool = False

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @classmethod
    def for_testing(cls, **overrides) -> Settings:
        """Ephemeral, file-free settings for tests and demos."""
        base = {
            "database_path": ":memory:",
            "api_keys_file": None,
            "audit_log_path": None,
            "log_json": False,
        }
        return cls(_env_file=None, **{**base, **overrides})
