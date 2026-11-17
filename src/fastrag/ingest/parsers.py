"""Turn files into sections of text with headings and page numbers.

The default parsers are light (pypdf, BeautifulSoup, python-docx). Install the
`docling` extra for layout-aware PDF and Office parsing; it is used
automatically when present and FASTRAG_PARSER=docling.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import PurePosixPath
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Section:
    text: str
    heading: list[str] = field(default_factory=list)
    page: int | None = None
    kind: str = "text"  # "text" or "table"


@dataclass
class ParsedDocument:
    title: str
    sections: list[Section]
    meta: dict[str, object] = field(default_factory=dict)
    scanned_pages: list[int] = field(default_factory=list)
    page_images: dict[int, list[bytes]] = field(default_factory=dict)


class UnsupportedFile(ValueError):
    pass


MIME_BY_EXT = {
    ".pdf": "application/pdf",
    ".html": "text/html",
    ".htm": "text/html",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",
    ".csv": "text/csv",
}


def guess_mime(filename: str, declared: str | None = None) -> str:
    ext = PurePosixPath(filename.lower()).suffix
    if ext in MIME_BY_EXT:
        return MIME_BY_EXT[ext]
    if declared and declared != "application/octet-stream":
        return declared.split(";")[0].strip()
    raise UnsupportedFile(f"unsupported file type: {filename}")


def _stem(filename: str) -> str:
    return PurePosixPath(filename).stem.replace("_", " ").strip() or filename


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1258", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


# Headings common in Vietnamese policies, contracts and brochures.
_VI_HEADING = re.compile(
    r"^(chương|mục|phần|điều|article|chapter|section)\s+[\dIVXLC]+[.:]?\s*.{0,120}$"
    r"|^([IVXLC]+|\d{1,2}(\.\d{1,2}){0,2})[.)]\s+\S.{0,100}$",
    re.IGNORECASE,
)


def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 140 or (s.endswith((".", ",", ";")) and not _VI_HEADING.match(s)):
        return False
    if _VI_HEADING.match(s):
        return True
    letters = [c for c in s if c.isalpha()]
    return len(letters) >= 4 and len(s) <= 90 and all(c.isupper() for c in letters)


def _heading_level(line: str) -> int:
    s = line.strip().lower()
    if s.startswith(("chương", "phần", "chapter")):
        return 1
    if s.startswith(("mục", "section")) or re.match(r"^[ivxlc]+[.)]", s):
        return 2
    if s.startswith(("điều", "article")):
        return 3
    m = re.match(r"^(\d{1,2}(?:\.\d{1,2}){0,2})[.)]", s)
    if m:
        return 2 + m.group(1).count(".")
    return 2


def _sections_from_lines(lines: list[tuple[str, int | None]]) -> list[Section]:
    """Group plain text lines into paragraphs under detected headings."""
    sections: list[Section] = []
    path: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_page: int | None = None

    def flush() -> None:
        nonlocal buf, buf_page
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(text, [h for _, h in path], buf_page))
        buf, buf_page = [], None

    for line, page in lines:
        if _looks_like_heading(line):
            flush()
            level = _heading_level(line)
            path = [(lvl, h) for lvl, h in path if lvl < level] + [(level, line.strip())]
            continue
        if not line.strip():
            if buf:
                buf.append("")
            continue
        if buf_page is None:
            buf_page = page
        buf.append(line)
    flush()
    return _split_paragraphs(sections)


def _split_paragraphs(sections: list[Section]) -> list[Section]:
    out: list[Section] = []
    for sec in sections:
        for para in re.split(r"\n\s*\n", sec.text):
            # Re-join lines broken by PDF layout inside a sentence.
            para = re.sub(r"(?<![.:;!?])\n(?=[a-zà-ỹđ0-9(])", " ", para.strip())
            if para:
                out.append(Section(para, sec.heading, sec.page, sec.kind))
    return out


def parse_pdf(data: bytes, filename: str) -> ParsedDocument:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    lines: list[tuple[str, int | None]] = []
    scanned: list[int] = []
    images: dict[int, list[bytes]] = {}
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # pragma: no cover - malformed pages
            logger.warning("pdf page %s of %s failed to extract", i, filename)
            text = ""
        if len(text.strip()) < 30:
            scanned.append(i)
            try:
                images[i] = [img.data for img in page.images][:4]
            except Exception:  # pragma: no cover
                images[i] = []
        for line in text.splitlines():
            lines.append((line, i))
        lines.append(("", i))
    title = _stem(filename)
    meta_title = (reader.metadata.title if reader.metadata else None) or ""
    if meta_title and len(meta_title) > 3 and not meta_title.lower().startswith(("microsoft", "untitled")):
        title = meta_title.strip()
    return ParsedDocument(
        title=title,
        sections=_sections_from_lines(lines),
        meta={"pages": len(reader.pages)},
        scanned_pages=scanned,
        page_images=images,
    )


def parse_html(data: bytes, filename: str) -> ParsedDocument:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_decode(data), "lxml")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "iframe", "svg"]):
        tag.decompose()
    title = (soup.title.string.strip() if soup.title and soup.title.string else "") or _stem(filename)
    body = soup.find("main") or soup.find("article") or soup.body or soup
    sections: list[Section] = []
    path: list[tuple[int, str]] = []
    for el in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table", "pre", "blockquote"]):
        if el.name and el.name[0] == "h" and el.name[1:].isdigit():
            level = int(el.name[1:])
            heading = el.get_text(" ", strip=True)
            if heading:
                path = [(lvl, h) for lvl, h in path if lvl < level] + [(level, heading)]
            continue
        if el.find_parent(["li", "table"]) and el.name != "table":
            continue  # counted with its parent
        if el.name == "table":
            rows = []
            for tr in el.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                sections.append(Section("\n".join(rows), [h for _, h in path], kind="table"))
            continue
        text = el.get_text(" ", strip=True)
        if text:
            sections.append(Section(("• " if el.name == "li" else "") + text, [h for _, h in path]))
    return ParsedDocument(title=title, sections=sections)


def parse_docx(data: bytes, filename: str) -> ParsedDocument:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(io.BytesIO(data))
    sections: list[Section] = []
    path: list[tuple[int, str]] = []
    title = (d.core_properties.title or "").strip() or _stem(filename)
    for block in d.element.body.iterchildren():
        tag = block.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(block, d)
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name or "").lower() if para.style is not None else ""
            m = re.match(r"(heading|title)\s*(\d*)", style)
            if m:
                level = int(m.group(2) or 1)
                path = [(lvl, h) for lvl, h in path if lvl < level] + [(level, text)]
                continue
            if _VI_HEADING.match(text) and len(text) < 120:
                level = _heading_level(text)
                path = [(lvl, h) for lvl, h in path if lvl < level] + [(level, text)]
                continue
            sections.append(Section(text, [h for _, h in path]))
        elif tag == "tbl":
            table = Table(block, d)
            rows = []
            for row in table.rows:
                cells: list[str] = []
                for c in row.cells:
                    t = c.text.strip()
                    if not cells or cells[-1] != t:  # merged cells repeat
                        cells.append(t)
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                sections.append(Section("\n".join(rows), [h for _, h in path], kind="table"))
    return ParsedDocument(title=title, sections=sections)


def parse_markdown(data: bytes, filename: str, detect_vi_headings: bool = False) -> ParsedDocument:
    """Markdown headings define sections. With detect_vi_headings, plain lines
    such as "Điều 4. Bàn giao" also start sections; docling output needs this
    because it does not always mark Vietnamese legal headings as headings."""
    text = _decode(data)
    level_by_kind: dict[str, int] = {}
    sections: list[Section] = []
    path: list[tuple[int, str]] = []
    buf: list[str] = []
    title = _stem(filename)

    def flush() -> None:
        if "".join(buf).strip():
            for para in re.split(r"\n\s*\n", "\n".join(buf)):
                if para.strip():
                    kind = "table" if para.strip().startswith("|") else "text"
                    sections.append(Section(para.strip(), [h for _, h in path], kind=kind))
        buf.clear()

    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.+?)\s*#*$", line)
        heading: tuple[int, str] | None = None
        if m:
            heading = (len(m.group(1)), m.group(2))
        elif detect_vi_headings and _VI_HEADING.match(line.strip()) and len(line.strip()) < 120:
            kind = line.strip().split()[0].lower()
            heading = (level_by_kind.get(kind, _heading_level(line) + 1), line.strip())
        if heading:
            flush()
            level, htext = heading
            level_by_kind.setdefault(htext.split()[0].lower(), level)
            if level == 1 and not sections and not path:
                title = htext
            path = [(lvl, h) for lvl, h in path if lvl < level] + [(level, htext)]
        else:
            buf.append(line)
    flush()
    return ParsedDocument(title=title, sections=sections)


def parse_text(data: bytes, filename: str) -> ParsedDocument:
    lines: list[tuple[str, int | None]] = [(line, None) for line in _decode(data).splitlines()]
    return ParsedDocument(title=_stem(filename), sections=_sections_from_lines(lines))


@lru_cache(maxsize=1)
def _docling_converter() -> Any:  # pragma: no cover - optional
    from docling.document_converter import DocumentConverter

    return DocumentConverter()  # loads layout models once per process


def _parse_docling(data: bytes, filename: str) -> ParsedDocument:  # pragma: no cover - optional
    import tempfile

    suffix = PurePosixPath(filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        result = _docling_converter().convert(tmp.name)
    md = result.document.export_to_markdown()
    parsed = parse_markdown(md.encode("utf-8"), filename, detect_vi_headings=True)
    if suffix.lower() == ".pdf":
        # Same title rule as the builtin parser: PDF metadata when it is meaningful.
        builtin = parse_pdf(data, filename)
        if builtin.title != _stem(filename):
            parsed.title = builtin.title
        parsed.meta, parsed.scanned_pages, parsed.page_images = builtin.meta, builtin.scanned_pages, builtin.page_images
    parsed.meta["parser"] = "docling"
    return parsed


def parse(data: bytes, filename: str, mime: str, prefer: str = "builtin") -> ParsedDocument:
    if prefer == "docling" and mime in {MIME_BY_EXT[".pdf"], MIME_BY_EXT[".docx"]}:
        try:
            return _parse_docling(data, filename)
        except ImportError:
            logger.warning("docling not installed; using builtin parser")
    if mime == "application/pdf":
        return parse_pdf(data, filename)
    if mime == "text/html":
        return parse_html(data, filename)
    if mime == MIME_BY_EXT[".docx"]:
        return parse_docx(data, filename)
    if mime == "text/markdown":
        return parse_markdown(data, filename)
    if mime == "text/plain":
        return parse_text(data, filename)
    raise UnsupportedFile(f"cannot parse {mime} as a document; spreadsheets go to a listings collection")
