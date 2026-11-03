"""Split text into overlapping chunks that respect paragraph and sentence boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class _Unit:
    text: str
    new_paragraph: bool  # joined to the previous unit with a blank line, else a space


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Pack text into chunks of at most ``chunk_size`` characters.

    Text is broken into paragraphs, then sentences, then words, only as far as needed
    to fit. Paragraph breaks and line breaks inside a paragraph are kept, so lists and
    tables stay readable. Each chunk starts with up to ``overlap`` characters of whole
    sentences from the end of the previous one, so a fact that straddles a boundary is
    still retrievable from a single chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[str] = []
    current: list[_Unit] = []
    length = 0
    for unit in _split_units(text, chunk_size):
        if current and length + _sep(unit) + len(unit.text) > chunk_size:
            chunks.append(_join(current))
            current = _overlap_tail(current, overlap)
            length = len(_join(current))
            if current and length + _sep(unit) + len(unit.text) > chunk_size:
                current, length = [], 0
        length += (_sep(unit) if current else 0) + len(unit.text)
        current.append(unit)
    if current:
        chunks.append(_join(current))
    return chunks


def _sep(unit: _Unit) -> int:
    return 2 if unit.new_paragraph else 1


def _join(units: list[_Unit]) -> str:
    parts: list[str] = []
    for i, unit in enumerate(units):
        if i:
            parts.append("\n\n" if unit.new_paragraph else " ")
        parts.append(unit.text)
    return "".join(parts)


def _normalize(paragraph: str) -> str:
    """Collapse runs of spaces within each line; keep the line breaks themselves."""
    lines = (" ".join(line.split()) for line in paragraph.splitlines())
    return "\n".join(line for line in lines if line)


def _split_units(text: str, max_len: int) -> list[_Unit]:
    """Break text into pieces no longer than ``max_len``, preferring natural boundaries."""
    units: list[_Unit] = []
    for paragraph in _PARAGRAPH_SPLIT.split(text):
        paragraph = _normalize(paragraph)
        if not paragraph:
            continue
        if len(paragraph) <= max_len:
            units.append(_Unit(paragraph, True))
            continue
        pieces: list[str] = []
        for sentence in _SENTENCE_SPLIT.split(paragraph):
            pieces.extend(
                [sentence] if len(sentence) <= max_len else _split_words(sentence, max_len)
            )
        units.extend(_Unit(piece, i == 0) for i, piece in enumerate(pieces))
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


def _overlap_tail(units: list[_Unit], overlap: int) -> list[_Unit]:
    """The trailing whole sentences of ``units`` that fit in ``overlap`` characters."""
    tail: list[_Unit] = []
    size = 0
    for unit in reversed(units):
        sentences = _SENTENCE_SPLIT.split(unit.text)
        taken: list[str] = []
        for sentence in reversed(sentences):
            added = len(sentence) + (1 if taken or tail else 0)
            if size + added > overlap:
                break
            taken.insert(0, sentence)
            size += added
        if taken:
            whole = len(taken) == len(sentences)
            tail.insert(0, _Unit(" ".join(taken), unit.new_paragraph and whole))
        if len(taken) != len(sentences):
            break  # stop at the first sentence that doesn't fit
    return tail
