"""PII redaction applied at ingest, before anything is embedded, indexed or sent to an LLM.

Redacting at the edge means PII never lands in the vector store, the keyword index, the
prompt or the cache. Regexes are a baseline; production systems typically add an NER
model (e.g. Microsoft Presidio) behind the same ``redact`` interface.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Quantifiers are bounded (RFC 5321 limits) so matching stays linear: an unbounded
# `[...]+@` re-scans the whole run from every start position, which is quadratic.
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]{1,64}@(?:[A-Za-z0-9-]{1,63}\.){1,8}[A-Za-z]{2,24}\b")
_SSN = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
_PHONE = re.compile(r"(?<![\w-])(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?![\w-])")


@dataclass(frozen=True)
class RedactionResult:
    text: str
    counts: dict[str, int]


class Redactor:
    def redact(self, text: str) -> RedactionResult:
        counts: Counter[str] = Counter()

        def sub(pattern: re.Pattern[str], label: str, text: str, check=None) -> str:
            def replace(match: re.Match[str]) -> str:
                if check is not None and not check(match.group(0)):
                    return match.group(0)
                counts[label] += 1
                return f"[REDACTED_{label}]"

            return pattern.sub(replace, text)

        text = sub(_EMAIL, "EMAIL", text)
        text = sub(_SSN, "SSN", text)
        text = sub(_CARD, "CARD", text, lambda s: _luhn_valid(re.sub(r"\D", "", s)))
        text = sub(_PHONE, "PHONE", text)
        return RedactionResult(text=text, counts=dict(counts))
