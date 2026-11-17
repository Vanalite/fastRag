"""OpenAI-compatible chat clients. Used only at ingestion (D13).

`FASTRAG_LLM_URL` / `FASTRAG_LLM_MODEL` define the provider named "default";
`FASTRAG_LLM_PROVIDERS` adds more by name, and each documents collection picks
one for its contextual summaries (D25).
"""

from __future__ import annotations

import os

import httpx

from fastrag.config import LlmProvider, Settings
from fastrag.providers.http import bearer, post_json

DEFAULT_PROVIDER = "default"


class ChatLLM:
    def __init__(self, name: str, provider: LlmProvider, api_key: str | None = None):
        self.name = name
        self.model = provider.model
        self.concurrency = provider.concurrency
        self.input_cost_per_mtok = provider.input_cost_per_mtok
        self.output_cost_per_mtok = provider.output_cost_per_mtok
        base = provider.url.rstrip("/")
        self.url = base if base.endswith("/chat/completions") else f"{base}/v1/chat/completions"
        self.headers = bearer(api_key)
        self.client = httpx.AsyncClient(timeout=provider.timeout_s)

    async def complete(self, system: str, user: str, max_tokens: int = 200) -> str:
        data = await post_json(
            self.client,
            self.url,
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            headers=self.headers,
        )
        return str(data["choices"][0]["message"]["content"]).strip()

    async def aclose(self) -> None:
        await self.client.aclose()


def build_llms(s: Settings) -> dict[str, ChatLLM]:
    """Every configured provider by name. Empty when no LLM is configured."""
    out: dict[str, ChatLLM] = {}
    if s.llm_url and s.llm_model:
        default = LlmProvider(
            url=s.llm_url,
            model=s.llm_model,
            timeout_s=s.llm_timeout_s,
            concurrency=s.llm_concurrency,
            input_cost_per_mtok=s.llm_input_cost_per_mtok,
            output_cost_per_mtok=s.llm_output_cost_per_mtok,
        )
        out[DEFAULT_PROVIDER] = ChatLLM(
            DEFAULT_PROVIDER, default, s.llm_api_key.get_secret_value() if s.llm_api_key else None
        )
    for name, provider in s.llm_providers.items():
        if name == DEFAULT_PROVIDER:
            raise ValueError('FASTRAG_LLM_PROVIDERS: the name "default" is reserved for FASTRAG_LLM_URL')
        key = os.environ.get(provider.api_key_env) if provider.api_key_env else None
        if provider.api_key_env and not key:
            raise ValueError(f"FASTRAG_LLM_PROVIDERS.{name}: environment variable {provider.api_key_env} is not set")
        out[name] = ChatLLM(name, provider, key)
    return out


def default_llm(llms: dict[str, ChatLLM]) -> ChatLLM | None:
    """The "default" provider, or the only one when exactly one is configured."""
    if DEFAULT_PROVIDER in llms:
        return llms[DEFAULT_PROVIDER]
    return next(iter(llms.values())) if len(llms) == 1 else None


def build_llm(s: Settings) -> ChatLLM | None:
    return default_llm(build_llms(s))
