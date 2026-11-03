from __future__ import annotations

from fastrag.domain import ScoredChunk
from fastrag.generation.base import Generation


class ExtractiveGenerator:
    """Offline generator: quotes the best passages with citations. No LLM, no API key."""

    name = "extractive"

    def __init__(self, max_passages: int = 3):
        self.max_passages = max_passages

    def generate(self, question: str, sources: list[ScoredChunk]) -> Generation:
        shown = sources[: self.max_passages]
        lines = ["Most relevant passages:"]
        for i, source in enumerate(shown, start=1):
            lines.append(f"\n[{i}] {source.chunk.text}")
        return Generation(text="\n".join(lines), citations=list(range(1, len(shown) + 1)))
