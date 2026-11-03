"""Grounded answer generation with Claude."""

from __future__ import annotations

from typing import Any

from fastrag.domain import ScoredChunk, Usage
from fastrag.errors import ConfigurationError, GenerationError
from fastrag.generation.base import Generation
from fastrag.generation.prompt import SYSTEM_PROMPT, build_user_prompt


class ClaudeGenerator:
    """Calls the Messages API.

    Credentials come from the environment (``ANTHROPIC_API_KEY``, or a profile from
    ``ant auth login``). Pass ``client`` to inject a preconfigured or fake client.
    """

    name = "claude"
    DEFAULT_MODEL = "claude-opus-5-5"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        max_tokens: int = 16000,
        client: Any | None = None,
    ):
        try:
            import anthropic

            self._api_errors: tuple[type[BaseException], ...] = (anthropic.APIError,)
        except ImportError:
            if client is None:
                raise ConfigurationError(
                    "The anthropic package is not installed; run: pip install 'fastrag[claude]'"
                ) from None
            self._api_errors = ()
            anthropic = None  # type: ignore[assignment]
        self._client = client if client is not None else anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def generate(self, question: str, sources: list[ScoredChunk]) -> Generation:
        try:
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                output_config={"effort": self.effort},
                # If a safety classifier declines, the API retries on a suitable fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                # The system prompt never changes, so it is cached; the per-question
                # documents come after it and don't break the cached prefix.
                system=[
                    {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
                ],
                messages=[{"role": "user", "content": build_user_prompt(question, sources)}],
            )
        except self._api_errors as exc:
            raise GenerationError(f"LLM request failed: {type(exc).__name__}") from exc

        if response.stop_reason == "refusal":
            raise GenerationError("The model declined to answer this question.")
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if response.stop_reason == "max_tokens":
            text += "\n\n[Answer truncated: reached max_tokens.]"

        usage = getattr(response, "usage", None)
        return Generation(
            text=text,
            usage=Usage(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            ),
        )
