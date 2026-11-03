"""Check that an answer only cites documents that were actually shown to the model."""

from __future__ import annotations

import re

_MARKER = re.compile(r"\[(\d{1,4})\]")


def validate_citations(text: str, source_count: int) -> tuple[str, list[int], list[int]]:
    """Return (cleaned_text, valid_indices_in_order, invalid_indices).

    Markers outside 1..source_count are removed from the text: a citation the reader
    can't follow is worse than none, and it often signals a hallucinated source.
    """
    used: list[int] = []
    invalid: list[int] = []

    def check(match: re.Match[str]) -> str:
        n = int(match.group(1))
        if 1 <= n <= source_count:
            if n not in used:
                used.append(n)
            return match.group(0)
        invalid.append(n)
        return ""

    cleaned = _MARKER.sub(check, text)
    if invalid:
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).replace(" .", ".").strip()
    return cleaned, used, invalid
