from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from fastrag.domain import ScoredChunk, Usage


@dataclass(frozen=True)
class Generation:
    text: str
    usage: Usage = field(default_factory=Usage)
    # Source indices (1-based) the answer cites, when the generator knows them exactly.
    # None means "parse [n] markers from the text".
    citations: list[int] | None = None


class Generator(Protocol):
    name: str

    def generate(self, question: str, sources: list[ScoredChunk]) -> Generation:
        """Answer ``question`` from ``sources`` only, citing them as [1], [2], ..."""
        ...
