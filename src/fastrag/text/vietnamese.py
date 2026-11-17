"""Vietnamese text handling (design log D5).

- Unicode NFC, plus repair of TCVN3 (ABC) text from old PDFs and fonts.
- One canonical tone-mark placement ("hòa", not "hoà") so both typing styles match.
- Accent-free forms for fallback matching.
- Word segmentation: spaces separate syllables, not words, so "chung cư" is one word.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)

# --- Legacy encodings -------------------------------------------------------

# TCVN3 (ABC) maps Vietnamese letters onto single Latin-1 code points. Text
# extracted from PDFs that used .VnTime fonts comes out like "Hîp ®ång".
_TCVN3 = {
    "µ": "à",
    "¸": "á",
    "¶": "ả",
    "·": "ã",
    "¹": "ạ",
    "¨": "ă",
    "»": "ằ",
    "¾": "ắ",
    "¼": "ẳ",
    "½": "ẵ",
    "Æ": "ặ",
    "©": "â",
    "Ç": "ầ",
    "Ê": "ấ",
    "È": "ẩ",
    "É": "ẫ",
    "Ë": "ậ",
    "®": "đ",
    "Ì": "è",
    "Ð": "é",
    "Î": "ẻ",
    "Ï": "ẽ",
    "Ñ": "ẹ",
    "ª": "ê",
    "Ò": "ề",
    "Õ": "ế",
    "Ó": "ể",
    "Ô": "ễ",
    "Ö": "ệ",
    "×": "ì",
    "Ý": "í",
    "Ø": "ỉ",
    "Ü": "ĩ",
    "Þ": "ị",
    "ß": "ò",
    "ã": "ó",
    "á": "ỏ",
    "â": "õ",
    "ä": "ọ",
    "«": "ô",
    "å": "ồ",
    "è": "ố",
    "æ": "ổ",
    "ç": "ỗ",
    "é": "ộ",
    "¬": "ơ",
    "ê": "ờ",
    "í": "ớ",
    "ë": "ở",
    "ì": "ỡ",
    "î": "ợ",
    "ï": "ù",
    "ó": "ú",
    "ñ": "ủ",
    "ò": "ũ",
    "ô": "ụ",
    "­": "ư",
    "õ": "ừ",
    "ø": "ứ",
    "ö": "ử",
    "÷": "ữ",
    "ù": "ự",
    "ú": "ỳ",
    "ý": "ý",
    "û": "ỷ",
    "ü": "ỹ",
    "þ": "ỵ",
    "¡": "Ă",
    "¢": "Â",
    "§": "Đ",
    "£": "Ê",
    "¤": "Ô",
    "¥": "Ơ",
    "¦": "Ư",
}
_TCVN3_TABLE = str.maketrans(_TCVN3)
# Code points that are common in TCVN3 text but almost never in real
# Vietnamese or Latin-1 prose.
_TCVN3_MARKERS = set("µ¸¶·¹¨»¾¼½Æ©ÇÊÈÉË®ÌÐÎÏÑªÒÕÓÔÖ×ÝØÜÞß«¬­")
# VNI Windows writes a base letter followed by mark characters, e.g. "Vieät".
_VNI_PATTERN = re.compile(r"[a-zA-Z](?:[øùûõï]|[âêô][øùûõï]?)\b|[a-zA-Z][àáåãäèéúüë]")

_VI_LETTERS = set("àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ")


@dataclass
class EncodingReport:
    encoding: str  # "unicode", "tcvn3" or "vni-suspected"
    converted: bool


def detect_legacy_encoding(text: str) -> str:
    """Return "tcvn3", "vni-suspected" or "unicode"."""
    sample = text[:20000]
    if not sample:
        return "unicode"
    letters = sum(ch.isalpha() for ch in sample) or 1
    markers = sum(ch in _TCVN3_MARKERS for ch in sample)
    real_vi = sum(ch.lower() in _VI_LETTERS for ch in sample)
    if markers / letters > 0.02 and markers > real_vi:
        return "tcvn3"
    vni_hits = len(_VNI_PATTERN.findall(sample))
    if vni_hits / letters > 0.03 and real_vi < vni_hits:
        return "vni-suspected"
    return "unicode"


def repair_encoding(text: str) -> tuple[str, EncodingReport]:
    """Convert TCVN3 to Unicode. VNI is only flagged: converting it wrongly
    would corrupt text, so suspected files go to review instead."""
    enc = detect_legacy_encoding(text)
    if enc == "tcvn3":
        return text.translate(_TCVN3_TABLE), EncodingReport("tcvn3", True)
    return text, EncodingReport(enc, False)


# --- Tone placement and diacritics -----------------------------------------

_TONE_MARKS = {"̀", "́", "̉", "̃", "̣"}  # huyền sắc hỏi ngã nặng

# "oa", "oe", "uy" without a final consonant: old style puts the tone on the
# first vowel ("hòa", "thủy"), new style on the second ("hoà", "thuỷ").
# We store old style because it is what most Vietnamese keyboards produce.
_NEW_STYLE = re.compile(r"(?<![qQgG])([oO])([aAeE])([̣̀́̉̃])(?![^\W\d_])")
_NEW_STYLE_UY = re.compile(r"(?<![qQ])([uU])([yY])([̣̀́̉̃])(?![^\W\d_])")


def normalize_tone_placement(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    decomposed = _NEW_STYLE.sub(lambda m: m.group(1) + m.group(3) + m.group(2), decomposed)
    decomposed = _NEW_STYLE_UY.sub(lambda m: m.group(1) + m.group(3) + m.group(2), decomposed)
    return unicodedata.normalize("NFC", decomposed)


def strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "D")


_WS = re.compile(r"[ \t  -​　]+")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f﻿]")


def normalize_text(text: str) -> tuple[str, EncodingReport]:
    """Ingest-time cleanup: legacy encodings, NFC, tone placement, whitespace."""
    text = _CONTROL.sub("", text)
    text, report = repair_encoding(text)
    text = unicodedata.normalize("NFC", text)
    text = normalize_tone_placement(text)
    text = _WS.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), report


def normalize_query(text: str) -> str:
    text = unicodedata.normalize("NFC", _CONTROL.sub("", text))
    return _WS.sub(" ", normalize_tone_placement(text)).strip()


# --- Segmentation -----------------------------------------------------------

_TOKEN = re.compile(r"[^\W_]+(?:[.,][0-9]+)*", re.UNICODE)


def _builtin_segment(text: str) -> list[str]:
    return _TOKEN.findall(text)


@lru_cache(maxsize=4)
def _load_segmenter(name: str) -> Callable[[str], list[str]]:
    if name == "underthesea":
        try:
            from underthesea import word_tokenize

            def seg(t: str) -> list[str]:
                return [w.replace(" ", "_") for w in word_tokenize(t)]

            seg("khởi động")
            return seg
        except Exception:  # pragma: no cover - depends on optional install
            logger.warning("underthesea unavailable, falling back to builtin segmenter")
    if name == "pyvi":
        try:
            from pyvi import ViTokenizer

            def seg_pyvi(t: str) -> list[str]:
                return ViTokenizer.tokenize(t).split()

            return seg_pyvi
        except Exception:  # pragma: no cover
            logger.warning("pyvi unavailable, falling back to builtin segmenter")
    return _builtin_segment


def segment(text: str, segmenter: str = "underthesea") -> list[str]:
    """Split text into words; multi-syllable words are joined with "_"."""
    seg = _load_segmenter(segmenter)
    words: list[str] = []
    # Segment per line: keeps the CRF input short and avoids joining across lines.
    for line in text.splitlines():
        if line.strip():
            words.extend(seg(line))
    return words


# Very common function words. "không" (not) is deliberately kept.
STOPWORDS = frozenset(
    """là của và các những có được cho với thì mà này đó một trong khi để ở
    ra vào lên về như từ theo đã sẽ đang bị bởi nên nhưng hay hoặc rằng
    the a an of and or to in on for is are""".split()
)

ACCENTLESS_PREFIX = "~"


@dataclass
class Terms:
    """Keyword-search terms for one text: term -> frequency."""

    freqs: dict[str, int] = field(default_factory=dict)
    length: int = 0


def keyword_terms(text: str, segmenter: str = "underthesea", extra: list[str] | None = None) -> Terms:
    """Terms for BM25. Each word is indexed in three ways:

    - the segmented word with diacritics ("chung_cư"), which ranks highest;
    - each syllable of a compound word ("chung", "cư"), in case the segmenter
      splits the query differently from the document;
    - an accent-free copy with a "~" prefix ("~chung_cu"), so questions typed
      without diacritics still match, at a lower weight.
    """
    terms = Terms()

    def add(term: str) -> None:
        terms.freqs[term] = terms.freqs.get(term, 0) + 1

    for raw in segment(text, segmenter):
        word = raw.lower().strip("_")
        if not word or not _TOKEN.search(word) or word.replace("_", " ") in STOPWORDS:
            continue
        terms.length += 1
        add(word)
        plain = strip_diacritics(word)
        add(ACCENTLESS_PREFIX + plain)
        if "_" in word:
            for syl, plain_syl in zip(word.split("_"), plain.split("_"), strict=False):
                if syl and syl not in STOPWORDS:
                    add(syl)
                    add(ACCENTLESS_PREFIX + plain_syl)
    for alias in extra or []:
        add(alias)
    return terms


def query_terms(text: str, segmenter: str = "underthesea", extra: list[str] | None = None) -> dict[str, int]:
    return keyword_terms(normalize_query(text), segmenter, extra).freqs


_PLAIN_STOPWORDS = frozenset(strip_diacritics(w) for w in STOPWORDS)
# Question words and particles carry no topic ("bao nhiêu", "gì", "phải không").
_QUESTION_WORDS = frozenset(
    "bao_nhiêu bao_lâu gì nào sao thế_nào như_thế_nào ở_đâu đâu ạ nhỉ vậy thế à hả ạ_ơi nhé "
    "phải_không phải_ko được_không có_không ko k hok hông chưa".split()
)
_TRAILING_PARTICLES = frozenset({"không", "chưa", "ko", "k", "hông", "hok"})


def query_words(text: str, segmenter: str = "underthesea") -> list[str]:
    """The content words of a question, for coverage and match signals.
    Words typed without diacritics that look like function words ("duoc",
    "la") are left out, since we cannot tell them from content words."""
    out: list[str] = []
    raw_words = [r.lower().strip("_") for r in segment(normalize_query(text), segmenter)]
    # Question phrases the segmenter left split, e.g. "bao" + "nhieu".
    plain_q = {strip_diacritics(q) for q in _QUESTION_WORDS if "_" in q}
    skip: set[int] = set()
    for i in range(len(raw_words) - 1):
        pair = f"{raw_words[i]}_{raw_words[i + 1]}"
        if pair in _QUESTION_WORDS or strip_diacritics(pair) in plain_q:
            skip.update((i, i + 1))
    for i, w in enumerate(raw_words):
        if i in skip:
            continue
        if not w or not _TOKEN.search(w) or w.replace("_", " ") in STOPWORDS:
            continue
        plain = strip_diacritics(w)
        if plain == w and plain.replace("_", " ") in _PLAIN_STOPWORDS:
            continue
        if w in _QUESTION_WORDS or plain in plain_q:
            continue
        if w not in out:
            out.append(w)
    # "... nuôi chó được không?": a trailing "không" asks, it does not negate.
    while out and out[-1] in _TRAILING_PARTICLES:
        out.pop()
    return out


_VI_ONLY = re.compile(r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]", re.IGNORECASE)


def is_accentless(text: str) -> bool:
    """A question of two or more words with no Vietnamese diacritics at all,
    e.g. "dat coc bao nhieu". English questions also count; the effect is
    only that word matches count as evidence alongside the reranker."""
    return not _VI_ONLY.search(text) and len(re.findall(r"[a-zA-Z]{2,}", text)) >= 2


def word_in_terms(word: str, terms: set[str]) -> bool:
    if word in terms:
        return True
    plain = strip_diacritics(word)
    if ACCENTLESS_PREFIX + plain in terms:
        return True
    parts = plain.split("_")
    return len(parts) > 1 and all(ACCENTLESS_PREFIX + p in terms for p in parts)
