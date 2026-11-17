"""Structure-aware chunking with token budgets.

Paragraphs are packed into chunks under one heading path until the target
size; long paragraphs are split at sentence ends; tables are split by rows
with the header row repeated. A small overlap carries the last sentence into
the next chunk.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from fastrag.ingest.parsers import Section


@dataclass
class ChunkDraft:
    text: str
    heading: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    tokens: int = 0


_SENTENCE = re.compile(r"(?<=[.!?…;:])\s+(?=[A-ZÀ-ỸĐ0-9\"“(•-])|\n+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s and s.strip()]


def _hard_split(text: str, max_tokens: int, count: Callable[[str], int]) -> list[str]:
    """Split a sentence with no usable boundary by words."""
    words, out, buf = text.split(), [], []
    for w in words:
        buf.append(w)
        if count(" ".join(buf)) >= max_tokens:
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return out


def _pieces(section: Section, max_tokens: int, count: Callable[[str], int]) -> list[str]:
    if count(section.text) <= max_tokens:
        return [section.text]
    if section.kind == "table":
        rows = section.text.split("\n")
        header, out, buf = rows[0], [], [rows[0]]
        for row in rows[1:]:
            if count("\n".join([*buf, row])) > max_tokens and len(buf) > 1:
                out.append("\n".join(buf))
                buf = [header]
            buf.append(row)
        out.append("\n".join(buf))
        return out
    pieces: list[str] = []
    for sent in split_sentences(section.text):
        pieces.extend([sent] if count(sent) <= max_tokens else _hard_split(sent, max_tokens, count))
    return pieces


def chunk_sections(
    sections: list[Section],
    count: Callable[[str], int],
    target: int = 350,
    maximum: int = 512,
    overlap: int = 40,
    strategy: str = "structure",
) -> list[ChunkDraft]:
    """`structure` (default) starts a new chunk at every heading change.
    `window` packs text across headings: a chunk keeps the heading path where it
    starts and later headings stay in its text as lines. Use it for documents
    split into many tiny sections (e.g. every line styled as a heading)."""
    if strategy not in ("structure", "window"):
        raise ValueError(f"unknown chunking strategy {strategy!r}")
    chunks: list[ChunkDraft] = []
    cur: list[str] = []
    cur_tokens = 0
    cur_heading: list[str] = []
    active_heading: list[str] = []
    pages: list[int] = []

    def emit(carry: bool) -> None:
        nonlocal cur, cur_tokens, pages, cur_heading
        if not cur:
            return
        text = "\n".join(cur).strip()
        chunks.append(
            ChunkDraft(
                text,
                list(cur_heading),
                min(pages) if pages else None,
                max(pages) if pages else None,
                count(text),
            )
        )
        tail = cur[-1] if carry and cur else ""
        if tail and count(tail) <= overlap and len(cur) > 1:
            cur, cur_tokens = [tail], count(tail)
        else:
            cur, cur_tokens = [], 0
        pages = pages[-1:] if cur else []
        cur_heading = active_heading

    for sec in sections:
        if strategy == "structure":
            if sec.heading != cur_heading:
                emit(carry=False)
                cur_heading = sec.heading
        elif sec.heading != active_heading:
            if not cur:
                cur_heading = sec.heading
            elif sec.heading:
                # Keep the heading text in the running chunk.
                line = sec.heading[-1]
                cur.append(line)
                cur_tokens += count(line)
        active_heading = sec.heading
        for piece in _pieces(sec, maximum, count):
            n = count(piece)
            if cur and cur_tokens + n > target:
                emit(carry=sec.kind != "table")
            if cur and cur_tokens + n > maximum:
                emit(carry=False)
            cur.append(piece)
            cur_tokens += n
            if sec.page is not None:
                pages.append(sec.page)
    emit(carry=False)
    # Merge a very small trailing chunk into its predecessor under the same heading.
    merged: list[ChunkDraft] = []
    for ch in chunks:
        if (
            merged
            and ch.tokens < target // 4
            and merged[-1].heading == ch.heading
            and merged[-1].tokens + ch.tokens <= maximum
        ):
            prev = merged[-1]
            prev.text = prev.text + "\n" + ch.text
            prev.tokens = count(prev.text)
            prev.page_end = ch.page_end or prev.page_end
        else:
            merged.append(ch)
    return merged
