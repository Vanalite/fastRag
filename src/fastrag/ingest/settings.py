"""Per-collection ingestion settings (D25).

A documents collection may override the deployment defaults under
`collections.settings.ingest`. Only the keys given are overridden:

    {"ingest": {
        "parser": "builtin" | "docling",
        "ocr": true,
        "chunking": {"strategy": "structure" | "window",
                     "target_tokens": 350, "max_tokens": 512, "overlap_tokens": 40},
        "context": {"mode": "auto" | "none" | "header" | "llm",
                    "llm": "<provider name>", "instructions": "...", "summary_max_tokens": 120}
    }}

Every document records the fingerprint of the settings it was built with, so
a settings change can re-process exactly the documents that are out of date.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fastrag.providers.llm import DEFAULT_PROVIDER, ChatLLM, default_llm

if TYPE_CHECKING:
    from fastrag.services import Services

# Bump when the ingestion code changes in a way that should re-process documents.
PIPELINE_VERSION = 1

SUMMARY_MAX_TOKENS = 120


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChunkingOverrides(_Strict):
    strategy: Literal["structure", "window"] | None = Field(
        default=None,
        description="structure: chunks never cross a heading (default). window: pack text across headings, "
        "for documents with many tiny sections.",
    )
    target_tokens: int | None = Field(default=None, ge=50, le=4000)
    max_tokens: int | None = Field(default=None, ge=64, le=8000)
    overlap_tokens: int | None = Field(default=None, ge=0, le=1000)


class ContextOverrides(_Strict):
    mode: Literal["auto", "none", "header", "llm"] | None = Field(
        default=None,
        description="auto: LLM summary when an LLM is configured, else header. none: chunk text only. "
        "header: title and heading path. llm: header plus an LLM summary (fails validation without an LLM).",
    )
    llm: str | None = Field(default=None, min_length=1, max_length=64, description="LLM provider name.")
    instructions: str | None = Field(
        default=None, max_length=1000, description="Extra guidance for the summary, e.g. 'Name the project'."
    )
    summary_max_tokens: int | None = Field(default=None, ge=20, le=400)


class IngestOverrides(_Strict):
    parser: Literal["builtin", "docling"] | None = None
    ocr: bool | None = Field(default=None, description="Send scanned pages to OCR when it is configured.")
    chunking: ChunkingOverrides | None = None
    context: ContextOverrides | None = None


class IngestSettingsError(ValueError):
    pass


@dataclass(frozen=True)
class IngestConfig:
    """Effective settings for one collection, with the provider resolved."""

    parser: str
    ocr: bool
    strategy: str
    target_tokens: int
    max_tokens: int
    overlap_tokens: int
    context_mode: str  # none | header | llm (resolved)
    llm_name: str | None
    llm_model: str | None
    instructions: str | None
    summary_max_tokens: int
    tokenizer: str
    embedding_model: str
    fingerprint: str
    warnings: tuple[str, ...] = field(default=())

    def describe(self) -> dict[str, Any]:
        out = asdict(self)
        out["warnings"] = list(self.warnings)
        out["pipeline_version"] = PIPELINE_VERSION
        return out


def parse_overrides(raw: Any) -> IngestOverrides:
    if raw is None:
        return IngestOverrides()
    try:
        return IngestOverrides.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"ingest.{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors(include_url=False)
        )
        raise IngestSettingsError(problems) from exc


def merge_settings(base: dict[str, Any] | None, patch: dict[str, Any] | None) -> dict[str, Any]:
    """Deep-merge `patch` into `base`. A None value removes the key."""
    out = dict(base or {})
    for key, value in (patch or {}).items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_settings(out[key], value)
        else:
            out[key] = value
    return out


def resolve_ingest(
    collection_settings: dict[str, Any] | None,
    services: Services,
    *,
    strict: bool = False,
) -> IngestConfig:
    """Effective settings for a collection. `strict` turns problems that would
    otherwise fall back to a safe default (a missing LLM provider) into errors;
    use it when an admin saves settings."""
    s = services.settings
    ov = parse_overrides((collection_settings or {}).get("ingest"))
    ch = ov.chunking or ChunkingOverrides()
    cx = ov.context or ContextOverrides()
    warnings: list[str] = []

    target = ch.target_tokens or s.chunk_target_tokens
    maximum = ch.max_tokens or s.chunk_max_tokens
    overlap = ch.overlap_tokens if ch.overlap_tokens is not None else s.chunk_overlap_tokens
    if target > maximum:
        raise IngestSettingsError(f"ingest.chunking: target_tokens ({target}) must not exceed max_tokens ({maximum})")
    if overlap >= target:
        raise IngestSettingsError(f"ingest.chunking: overlap_tokens ({overlap}) must be below target_tokens ({target})")

    parser = ov.parser or s.parser
    if strict and parser == "docling" and importlib.util.find_spec("docling") is None:
        raise IngestSettingsError("ingest.parser: docling is not installed (install the `docling` extra)")

    mode = cx.mode or "auto"
    llm = _pick_llm(services, cx.llm)
    if cx.llm and llm is None:
        message = f"ingest.context.llm: no LLM provider named {cx.llm!r}; configured: {_names(services)}"
        if strict:
            raise IngestSettingsError(message)
        warnings.append(message + "; using header context")
    if mode == "llm" and llm is None:
        if strict:
            raise IngestSettingsError("ingest.context.mode is 'llm' but no LLM is configured (FASTRAG_LLM_URL)")
        if not cx.llm:
            warnings.append("context mode 'llm' but no LLM is configured; using header context")
        mode = "header"
    elif mode == "auto":
        mode = "llm" if llm is not None else "header"
    if mode != "llm":
        llm = None

    instructions = (cx.instructions or "").strip() or None
    values: dict[str, Any] = {
        "parser": parser,
        "ocr": True if ov.ocr is None else ov.ocr,
        "strategy": ch.strategy or "structure",
        "target_tokens": target,
        "max_tokens": maximum,
        "overlap_tokens": overlap,
        "context_mode": mode,
        "llm_name": llm.name if llm else None,
        "llm_model": llm.model if llm else None,
        "instructions": instructions if mode == "llm" else None,
        "summary_max_tokens": cx.summary_max_tokens or SUMMARY_MAX_TOKENS,
        "tokenizer": s.tokenizer,
        "embedding_model": services.embedder.model,
    }
    digest = hashlib.sha256(json.dumps({"pipeline": PIPELINE_VERSION, **values}, sort_keys=True).encode()).hexdigest()
    return IngestConfig(**values, fingerprint=digest[:16], warnings=tuple(warnings))


def llm_for(services: Services, cfg: IngestConfig) -> ChatLLM | None:
    if cfg.context_mode != "llm" or cfg.llm_name is None:
        return None
    return _all_llms(services).get(cfg.llm_name)


def validate_collection_settings(collection_type: str, settings: dict[str, Any], services: Services) -> None:
    """Called when an admin creates or updates a collection."""
    if "ingest" not in settings:
        return
    if collection_type != "documents":
        raise IngestSettingsError("ingest settings apply to documents collections only")
    resolve_ingest(settings, services, strict=True)


def _all_llms(services: Services) -> dict[str, ChatLLM]:
    llms = dict(services.llms)
    if services.llm is not None:
        llms.setdefault(services.llm.name, services.llm)
    return llms


def _pick_llm(services: Services, name: str | None) -> ChatLLM | None:
    llms = _all_llms(services)
    if name:
        return llms.get(name)
    return default_llm(llms) if DEFAULT_PROVIDER in llms or len(llms) == 1 else None


def _names(services: Services) -> str:
    return ", ".join(sorted(_all_llms(services))) or "none"
