import pytest

from fastrag.ingest.chunking import chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.", chunk_size=100, overlap=10) == ["Hello world."]


def test_empty_text_has_no_chunks():
    assert chunk_text("   \n\n  ", chunk_size=100, overlap=10) == []


def test_chunks_respect_size_limit():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(20))
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_consecutive_chunks_overlap():
    sentences = [f"Sentence number {i} is here." for i in range(30)]
    chunks = chunk_text(" ".join(sentences), chunk_size=120, overlap=60)
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        last_sentence_of_prev = prev.split(". ")[-1]
        assert last_sentence_of_prev.rstrip(".") in nxt


def test_no_content_is_lost():
    words = [f"w{i}" for i in range(500)]
    chunks = chunk_text(" ".join(words), chunk_size=50, overlap=0)
    assert " ".join(chunks).split() == words


def test_very_long_word_is_hard_split():
    chunks = chunk_text("x" * 250, chunk_size=100, overlap=0)
    assert chunks == ["x" * 100, "x" * 100, "x" * 50]


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_arguments(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=size, overlap=overlap)
