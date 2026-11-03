from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Turns text into L2-normalized float32 vectors."""

    @property
    def name(self) -> str:
        """Spec string that ``get_embedder`` can rebuild this embedder from."""
        ...

    @property
    def dim(self) -> int: ...

    # Optional class attribute: the cosine threshold above which a match counts as
    # evidence (see Retriever.supports). Embedders without one get 0.35.
    # default_min_score: float

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape ``(len(texts), dim)``."""
        ...
