"""Settings, from ``FASTRAG_*`` environment variables or a ``.env`` file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FASTRAG_",
        env_file=".env",
        extra="ignore",
        # Validation errors must never print raw input: it can contain secrets.
        hide_input_in_errors=True,
    )

    # "production" turns configuration mistakes into startup errors (see _check_production).
    environment: Literal["development", "production"] = "development"

    # Storage. SQLite (the default) is single-node: each process keeps its own in-memory
    # indexes. Set postgres_dsn to store documents, vectors and the keyword index in
    # Postgres + pgvector instead, so every replica queries the same data.
    database_path: str = ".fastrag/fastrag.db"  # ":memory:" for an ephemeral store
    postgres_dsn: SecretStr | None = None  # e.g. postgresql://fastrag:...@db:5432/fastrag
    postgres_pool_size: int = Field(10, ge=1, le=200)
    # HNSW candidate list size per query: higher = better recall, slower.
    postgres_ef_search: int = Field(100, ge=10, le=1000)
    # Apply schema migrations at startup. Turn off in production to run the app with a
    # role that can't change the schema; run `fastrag db migrate` during deploys instead.
    postgres_migrate_on_start: bool = True
    embedder: str = "hashing"

    # Ingestion
    chunk_size: int = Field(800, ge=100, le=8000)
    chunk_overlap: int = Field(100, ge=0)
    redact_pii: bool = True
    # Redact PII in questions before they are sent to the LLM (retrieval stays local and
    # uses the original question).
    redact_questions: bool = True
    max_document_chars: int = Field(2_000_000, ge=1)
    # Asynchronous ingestion jobs (POST /v1/ingest-jobs).
    ingest_workers: int = Field(2, ge=1, le=64)
    max_active_jobs_per_tenant: int = Field(10, ge=1)
    max_documents_per_job: int = Field(500, ge=1, le=10_000)
    # Bounds on what one replica holds in memory for queued jobs.
    max_queued_jobs: int = Field(100, ge=1)
    max_queued_chars: int = Field(100_000_000, ge=1)
    # Pace of background ingestion per tenant (documents); workers wait, never fail.
    job_documents_per_minute: float = Field(600, ge=0)
    job_documents_burst: int = Field(50, ge=0)

    # Retrieval
    top_k: int = Field(4, ge=1, le=20)
    candidate_pool: int = Field(20, ge=1, le=200)
    rrf_k: int = Field(60, ge=1)
    min_dense_score: float | None = Field(None, ge=-1, le=1)  # None: embedder default
    min_keyword_coverage: float = Field(0.3, ge=0, le=1)
    # Optional second pass over the top `rerank_pool` candidates, e.g. "cross-encoder" or
    # "cross-encoder:<model>" (needs the `st` extra).
    reranker: str | None = None
    rerank_pool: int = Field(20, ge=1, le=200)

    # Generation
    generator: Literal["extractive", "claude"] = "extractive"
    claude_model: str = "claude-opus-5-5"
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_question_chars: int = Field(2000, ge=1)
    drop_suspicious_chunks: bool = False

    # Security
    api_keys_file: Path | None = Path(".fastrag/api_keys.json")
    # Bearer tokens. Issuer and audience apply to both the shared-secret (HS256) and
    # the OIDC verifier; OIDC requires both.
    jwt_secret: SecretStr | None = None
    jwt_issuer: str | None = None
    jwt_audience: str | None = None
    oidc_jwks_url: str | None = None
    token_tenant_claim: str = "tenant"
    token_groups_claim: str = "groups"
    token_roles_claim: str = "roles"

    # Operations
    audit_log_path: Path | None = Path(".fastrag/audit.jsonl")
    audit_store_query_text: bool = False
    cache_ttl_seconds: float = Field(300, ge=0)
    cache_max_entries: int = Field(1000, ge=0)
    # Shared state for several replicas: rate-limit buckets and the answer cache live in
    # Redis when set (e.g. rediss://:password@redis:6380/0). Unset: in-process.
    redis_url: SecretStr | None = None
    rate_limit_per_minute: float = Field(120, ge=0)  # per tenant
    rate_limit_burst: int = Field(30, ge=0)
    user_rate_limit_per_minute: float = Field(60, ge=0)  # per user, within the tenant budget
    user_rate_limit_burst: int = Field(15, ge=0)
    auth_failures_per_minute: float = Field(30, ge=0)  # per client address
    auth_failure_burst: int = Field(10, ge=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    # /metrics is unauthenticated and labels series by tenant: enable it only where the
    # port is reachable from your monitoring network alone.
    expose_metrics: bool = False

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.oidc_jwks_url and not (self.jwt_issuer and self.jwt_audience):
            raise ValueError("oidc_jwks_url requires jwt_issuer and jwt_audience")
        if self.environment == "production":
            self._check_production()
        return self

    def _check_production(self) -> None:
        problems = []
        if self.postgres_dsn is None and self.database_path == ":memory:":
            problems.append("database_path must be a file, not :memory:")
        if self.audit_log_path is None:
            problems.append("audit_log_path must be set")
        if not (self.api_keys_file or self.jwt_secret or self.oidc_jwks_url):
            problems.append("configure at least one authentication method")
        if self.jwt_secret and not (self.jwt_issuer and self.jwt_audience):
            problems.append("jwt_issuer and jwt_audience are required with jwt_secret")
        if not self.log_json:
            problems.append("log_json must be true (logs are shipped as JSON)")
        if self.rate_limit_per_minute <= 0 or self.auth_failures_per_minute <= 0:
            problems.append("rate limits must be enabled")
        if problems:
            raise ValueError("unsafe production configuration: " + "; ".join(problems))

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
