"""Turn raw content into plain text plus a title. One function per media type."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

from fastrag.errors import InvalidInput

SUPPORTED_MEDIA_TYPES = ("text/plain", "text/markdown", "text/html")


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    text: str


def parse(content: str, media_type: str = "text/plain") -> ParsedDocument:
    media_type = media_type.split(";")[0].strip().lower()
    if media_type == "text/plain":
        return ParsedDocument(title="", text=content)
    if media_type == "text/markdown":
        return _parse_markdown(content)
    if media_type == "text/html":
        return _parse_html(content)
    raise InvalidInput(
        f"unsupported media type {media_type!r}; supported: {', '.join(SUPPORTED_MEDIA_TYPES)}"
    )


_MD_HEADING = re.compile(r"^\s{0,3}#\s+(.+?)\s*#*\s*$", re.MULTILINE)


def _parse_markdown(content: str) -> ParsedDocument:
    match = _MD_HEADING.search(content)
    return ParsedDocument(title=match.group(1) if match else "", text=content)


class _HTMLText(HTMLParser):
    _SKIP = {"script", "style", "noscript", "template", "svg"}
    _BLOCK = {
        "p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article",
        "h1", "h2", "h3", "h4", "h5", "h6", "header", "footer", "pre", "blockquote",
    }  # fmt: skip
    _INLINE_BREAK = {"td", "th", "dt", "dd"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self._BLOCK:
            self.parts.append("\n\n")
        elif tag in self._INLINE_BREAK:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self._BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        else:
            self.parts.append(data)


def _parse_html(content: str) -> ParsedDocument:
    parser = _HTMLText()
    parser.feed(content)
    parser.close()
    text = re.sub(r"[ \t]+", " ", "".join(parser.parts))
    text = re.sub(r"\n\s*\n\s*", "\n\n", text).strip()
    return ParsedDocument(title=" ".join("".join(parser.title_parts).split()), text=text)
