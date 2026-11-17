"""Ingestion hygiene (D3, D12): flag prompt injection, personal data and
internal-only business terms. Scanning only flags; the retrieval profile
decides what flagged content a caller may see, and an admin clears flags
after review.
"""

from __future__ import annotations

import re

from fastrag.text.vietnamese import strip_diacritics

_INJECTION = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (all |the )?(previous|prior|above)",
    r"you are now (a|an|the)\b",
    r"(reveal|print|show) (your|the) (system )?prompt",
    r"system prompt",
    r"<\|(im_start|im_end|system|endoftext)\|>",
    r"\[/?(inst|sys)\]",
    r"bo qua (tat ca )?(cac )?(huong dan|chi dan|lenh) (truoc|tren)",
    r"(tiet lo|in ra) (prompt|chi dan he thong)",
    r"ban (bay gio|hien gio) la (mot )?(tro ly|ai|chatbot)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION), re.IGNORECASE)

_PHONE = re.compile(r"(?<!\d)(?:\+?84|0)(?:[\s.-]?\d){9,10}(?!\d)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# CCCD (12 digits) / old CMND (9 digits) when labelled as such.
_ID_NUMBER = re.compile(r"(cccd|cmnd|can cuoc|chung minh)[^\d]{0,20}\d{9,12}", re.IGNORECASE)
_BANK = re.compile(r"(stk|so tai khoan|tai khoan ngan hang)[^\d]{0,20}\d{6,20}", re.IGNORECASE)

# Business terms that must not reach customers (D12 safety net).
_INTERNAL_TERMS = re.compile(
    r"gia chot|gia san|gia goc|hoa hong|chiet khau noi bo|noi bo|commission|"
    r"chi tieu doanh so|bang luong|luong thuong|kpi|khong chia se|confidential|internal only",
    re.IGNORECASE,
)

FLAG_INJECTION = "prompt_injection"
FLAG_SENSITIVE_PUBLIC = "sensitive_in_public"
QUARANTINE_FLAGS = (FLAG_INJECTION, FLAG_SENSITIVE_PUBLIC)


def scan(text: str, groups: list[str]) -> list[str]:
    """Flags for one chunk or cell. `groups` is where the content will be visible."""
    flags: list[str] = []
    plain = strip_diacritics(text).lower()
    if _INJECTION_RE.search(plain) or _INJECTION_RE.search(text):
        flags.append(FLAG_INJECTION)
    pii = []
    if _PHONE.search(text):
        pii.append("pii:phone")
    if _EMAIL.search(text):
        pii.append("pii:email")
    if _ID_NUMBER.search(plain):
        pii.append("pii:id_number")
    if _BANK.search(plain):
        pii.append("pii:bank_account")
    flags.extend(pii)
    if "public" in groups and (pii or _INTERNAL_TERMS.search(plain)):
        flags.append(FLAG_SENSITIVE_PUBLIC)
    return flags
