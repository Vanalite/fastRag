"""Answer generation: grounded prompt -> LLM (or extractive fallback) -> citation checks."""

from fastrag.generation.base import Generation, Generator
from fastrag.generation.citations import validate_citations
from fastrag.generation.claude import ClaudeGenerator
from fastrag.generation.extractive import ExtractiveGenerator
from fastrag.generation.prompt import SYSTEM_PROMPT, build_user_prompt, looks_like_injection

__all__ = [
    "SYSTEM_PROMPT",
    "ClaudeGenerator",
    "ExtractiveGenerator",
    "Generation",
    "Generator",
    "build_user_prompt",
    "looks_like_injection",
    "validate_citations",
]
