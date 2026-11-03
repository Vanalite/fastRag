"""Feature-hashing embedder: fast, deterministic, and needs no model download.

It matches on shared words and word pairs, not meaning. That is enough for
keyword-style retrieval and for tests; switch to ``st:<model>`` for semantic search.
"""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np

_TOKEN = re.compile(r"\w+", re.UNICODE)


class HashingEmbedder:
    # Feature-hashing cosines are low even for relevant text. Calibrated on eval/golden:
    # answerable questions reach 0.09-0.32, out-of-scope ones 0.07-0.21.
    default_min_score = 0.25

    def __init__(self, dim: int = 1024):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def name(self) -> str:
        return f"hashing:{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _TOKEN.findall(text.lower())
            features = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False)]
            counts: dict[int, float] = {}
            for feature in features:
                # Python's built-in hash() is salted per process, so use a stable hash.
                h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest())
                index = h % self._dim
                sign = 1.0 if (h >> 63) & 1 else -1.0
                counts[index] = counts.get(index, 0.0) + sign
            for index, value in counts.items():
                # Sublinear term frequency so repeated words don't dominate.
                out[row, index] = math.copysign(1.0 + math.log(abs(value)), value) if value else 0.0
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out
