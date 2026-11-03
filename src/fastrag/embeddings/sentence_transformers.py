from __future__ import annotations

import numpy as np


class SentenceTransformerEmbedder:
    # A common starting point for MiniLM-style models; recalibrate with `fastrag eval`.
    default_min_score = 0.35

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed; run: pip install 'fastrag[st]'"
            ) from exc
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def name(self) -> str:
        return f"st:{self._model_name}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32).reshape(len(texts), self._dim)
