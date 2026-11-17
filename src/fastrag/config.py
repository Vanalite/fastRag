"""Settings, read from environment variables (prefix FASTRAG_) or a .env file.

Defaults are chosen so `docker compose up` works with no edits. Only the
service tokens must be changed before going to production.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LlmProvider(BaseModel):
    """A named OpenAI-compatible chat endpoint that a collection can pick for
    its contextual summaries (D25). The key is read from `api_key_env`, so it
    never appears in the provider list itself."""

    model_config = ConfigDict(extra="forbid")

    url: str
    model: str
    api_key_env: str | None = None
    timeout_s: float = Field(default=60.0, gt=0)
    concurrency: int = Field(default=4, ge=1, le=64)
    # Prices in USD per million tokens, used only for the import dry-run estimate.
    input_cost_per_mtok: float | None = Field(default=None, ge=0)
    output_cost_per_mtok: float | None = Field(default=None, ge=0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FASTRAG_", env_file=".env", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = True

    # Database
    database_url: str = "postgresql+asyncpg://fastrag:fastrag@localhost:5432/fastrag"
    db_pool_size: int = 10
    db_max_overflow: int = 10
    # Postgres JIT compiles queries with high cost estimates; for fastRag's short
    # vector and BM25 queries that costs 10x the query itself (D22).
    db_jit: bool = False

    # Admin token for tenant/collection management. Per-tenant API keys are
    # stored hashed in the database and created with `fastrag create-key`.
    admin_token: SecretStr = SecretStr("change-me-admin")

    # Embeddings (D14: BGE-M3, open weights). The dimension is fixed when the
    # database is created; changing the model means re-indexing.
    embedding_provider: Literal["tei", "openai", "local", "hash"] = "tei"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_url: str = "http://localhost:8081"
    embedding_api_key: SecretStr | None = None
    embedding_batch_size: int = 16
    embedding_timeout_s: float = 60.0

    # Reranker (D14: hosted multilingual reranker, swappable without re-indexing).
    reranker_provider: Literal["tei", "cohere", "jina", "local", "none"] = "tei"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_url: str = "http://localhost:8082"
    reranker_api_key: SecretStr | None = None
    reranker_timeout_s: float = 2.0
    # Platt scaling "a,b": calibrated = sigmoid(a * logit(raw) + b). Leave unset
    # for the built-in default of the configured model; fit your own with
    # `fastrag calibrate` on the reference questions.
    reranker_calibration: str | None = None

    # LLM, used only at ingestion (D13): contextual chunk summaries and
    # spreadsheet header suggestions. Any OpenAI-compatible endpoint (LiteLLM,
    # vLLM, Ollama, OpenAI). Leave unset to use deterministic context only.
    llm_url: str | None = None
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_timeout_s: float = 60.0
    llm_concurrency: int = 4
    llm_input_cost_per_mtok: float | None = None
    llm_output_cost_per_mtok: float | None = None
    # More endpoints by name, as JSON: {"cheap": {"url": "...", "model": "...",
    # "api_key_env": "CHEAP_LLM_KEY"}}. The one above is named "default".
    # A collection chooses one with settings.ingest.context.llm.
    llm_providers: dict[str, LlmProvider] = Field(default_factory=dict)

    # Raw file storage
    storage_backend: Literal["local", "s3"] = "local"
    storage_path: str = "./data/files"
    s3_bucket: str | None = None
    s3_endpoint_url: str | None = None
    s3_prefix: str = "fastrag/"

    # Parsing. "docling" needs the `docling` extra.
    parser: Literal["builtin", "docling"] = "builtin"
    # OCR for scanned PDF pages (D5: not Tesseract). "vlm" sends page images to
    # an OpenAI-compatible vision model; "none" sends the document to review.
    ocr_provider: Literal["none", "vlm"] = "none"
    ocr_url: str | None = None
    ocr_model: str | None = None
    ocr_api_key: SecretStr | None = None

    # Vietnamese processing
    segmenter: Literal["underthesea", "pyvi", "builtin"] = "underthesea"

    # Chunking defaults (token budgets are measured in model tokens, see D5).
    # A documents collection can override these, the parser and the context
    # mode under settings.ingest (D25).
    # tokenizer: "heuristic", or a path / Hugging Face id of a tokenizer.json
    # (e.g. "BAAI/bge-m3") for exact counts; needs the `tokenizers` package.
    tokenizer: str = "heuristic"
    chunk_target_tokens: int = 350
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 40

    # Worker
    worker_poll_interval_s: float = 2.0
    worker_concurrency: int = 2
    job_max_attempts: int = 3
    scheduler_interval_s: float = 60.0

    # Local-folder sources may only read below this directory.
    local_sources_root: str = "./data/sources"
    # Web-page sources: block private and link-local addresses (SSRF guard).
    allow_private_urls: bool = False
    google_credentials_file: str | None = None

    # Search logs feed the reference-question set (Q12). Questions may contain
    # personal data, so they are purged after the retention period.
    search_log_enabled: bool = True
    search_log_retention_days: int = 90

    # Upload limits
    max_upload_mb: int = 50

    cors_origins: list[str] = Field(default_factory=list)


@lru_cache
def get_settings() -> Settings:
    return Settings()
