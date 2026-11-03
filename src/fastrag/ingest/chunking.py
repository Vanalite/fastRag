"""Split text into overlapping chunks that respect paragraph and sentence boundaries."""

from __future__ import annotations

import re

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Pack text into chunks of at most ``chunk_size`` characters.

    Text is broken into paragraphs, then sentences, then words, only as far as needed
    to fit. Consecutive chunks share up to ``overlap`` characters of trailing units so
    a fact split across a boundary is still retrievable.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for unit in _split_units(text, chunk_size):
        if current and length + 1 + len(unit) > chunk_size:
            chunks.append(" ".join(current))
            current = _overlap_tail(current, overlap)
            length = _joined_length(current)
            if current and length + 1 + len(unit) > chunk_size:
                current, length = [], 0
        length += len(unit) + (1 if current else 0)
        current.append(unit)

    if current:
        chunks.append(" ".join(current))
    return chunks


def _split_units(text: str, max_len: int) -> list[str]:
    """Break text into pieces no longer than ``max_len``, preferring natural boundaries."""
    units: list[str] = []
    for paragraph in _PARAGRAPH_SPLIT.split(text):
        paragraph = _WHITESPACE.sub(" ", paragraph).strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_len:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_SPLIT.split(paragraph):
            if len(sentence) <= max_len:
                units.append(sentence)
            else:
                units.extend(_split_words(sentence, max_len))
    return units


def _split_words(sentence: str, max_len: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for word in sentence.split(" "):
        # A single word longer than max_len gets hard-sliced.
        while len(word) > max_len:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:max_len])
            word = word[max_len:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) > max_len:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _overlap_tail(units: list[str], overlap: int) -> list[str]:
    tail: list[str] = []
    size = 0
    for unit in reversed(units):
        added = len(unit) + (1 if tail else 0)
        if size + added > overlap:
            break
        tail.insert(0, unit)
        size += added
    return tail


def _joined_length(units: list[str]) -> int:
    return sum(len(u) for u in units) + max(len(units) - 1, 0)
