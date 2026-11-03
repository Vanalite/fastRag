import numpy as np
import pytest

from fastrag.embeddings import HashingEmbedder, get_embedder


def test_shape_and_normalization():
    emb = HashingEmbedder(dim=64)
    vectors = emb.embed(["the quick brown fox", "lazy dog", ""])
    assert vectors.shape == (3, 64)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors[:2], axis=1), 1.0, atol=1e-5)
    assert not vectors[2].any()  # empty text -> zero vector


def test_deterministic_across_instances():
    a = HashingEmbedder().embed(["retrieval augmented generation"])
    b = HashingEmbedder().embed(["retrieval augmented generation"])
    assert np.array_equal(a, b)


def test_similar_texts_score_higher():
    emb = HashingEmbedder()
    q, near, far = emb.embed(
        ["how do cats sleep", "cats sleep for many hours a day", "stock market prices fell"]
    )
    assert q @ near > q @ far


def test_get_embedder_specs():
    assert get_embedder("hashing").dim == 1024
    assert get_embedder("hashing:256").name == "hashing:256"
    with pytest.raises(ValueError):
        get_embedder("nope")
