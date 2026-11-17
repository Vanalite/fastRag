"""Deterministic value standardization (design log D5, D13).

Prices, areas, districts, wards, bedrooms and directions are parsed the same
way at ingestion (spreadsheet cells) and at query time (filter arguments).
Anything that cannot be parsed raises ValueError: fastRag rejects a filter it
does not understand instead of guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from fastrag.text.vietnamese import normalize_query, strip_diacritics


def _fold(text: str) -> str:
    """Lowercase, accent-free, single-spaced, punctuation turned into spaces."""
    text = strip_diacritics(normalize_query(str(text))).lower()
    text = re.sub(r"[^\w]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# --- Numbers ----------------------------------------------------------------

_THOUSANDS = re.compile(r"^\d{1,3}(?:([.,])\d{3})(?:\1\d{3})*$")


def parse_number(text: str) -> float:
    """Vietnamese numbers: "3,5" = 3.5, "1.200" = 1200, "3,500,000" = 3500000."""
    s = text.strip().replace(" ", "")
    if not s:
        raise ValueError("empty number")
    if _THOUSANDS.match(s):
        # "1.200" or "3,500" read as thousands; a single group of three digits
        # after a separator is far more often a thousands group in listings.
        return float(re.sub(r"[.,]", "", s))
    if s.count(",") == 1 and "." not in s:
        s = s.replace(",", ".")
    elif "," in s and "." in s:
        # "1.234,5" (VN) or "1,234.5" (EN): the last separator is the decimal.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    try:
        return float(s)
    except ValueError as exc:
        raise ValueError(f"not a number: {text!r}") from exc


# --- Prices -----------------------------------------------------------------

_UNITS = {
    "ty": 1_000_000_000,
    "ti": 1_000_000_000,
    "b": 1_000_000_000,
    "bn": 1_000_000_000,
    "billion": 1_000_000_000,
    "trieu": 1_000_000,
    "tr": 1_000_000,
    "m": 1_000_000,
    "million": 1_000_000,
    "nghin": 1_000,
    "ngan": 1_000,
    "k": 1_000,
}
_NUM = r"\d+(?:[.,]\d+)*"
_PRICE_PART = re.compile(rf"({_NUM})\s*(ty|ti|trieu|tr|nghin|ngan|billion|million|bn|b|m|k)?(?![a-z])")
_PER_M2 = re.compile(r"(/|tren|moi|per)\s*(m2|met vuong|m)\b|\bm2\b$")
_NEGOTIABLE = {"thoa thuan", "lien he", "call", "contact", "tt"}


@dataclass(frozen=True)
class Price:
    vnd: int
    per_m2: bool = False


def parse_price(value: Any) -> Price:
    """Parse "3,5 tỷ", "3 tỷ 200", "850tr", "35 triệu/m2", "3.500.000.000 đ"."""
    if isinstance(value, int | float):
        if value <= 0:
            raise ValueError("price must be positive")
        # Bare numbers below 100,000 are almost always in tỷ or triệu, which we cannot tell apart.
        if value < 100_000:
            raise ValueError(f"ambiguous price {value!r}: add a unit (tỷ, triệu) or give VND")
        return Price(round(value))
    text = _fold(value).replace("vnd", "").replace("dong", "")
    text = re.sub(r"\bd\b", "", text).strip()
    if not text:
        raise ValueError("empty price")
    if text in _NEGOTIABLE:
        raise ValueError(f"price is not a number: {value!r}")
    # The fold step turned "3,5" into "3 5"; parse the original separators instead.
    raw = strip_diacritics(normalize_query(str(value))).lower()
    raw = raw.replace("vnd", "").replace("đ", "")
    per_m2 = bool(_PER_M2.search(_fold(raw)))
    raw = re.sub(r"(/|tren|moi|per)\s*(m2|m²|met vuong|m)\b", "", raw)
    parts = _PRICE_PART.findall(raw)
    if not parts:
        raise ValueError(f"cannot read price {value!r}")
    total = 0.0
    last_unit: int | None = None
    for i, (num, unit) in enumerate(parts):
        if unit:
            last_unit = _UNITS[unit]
            total += parse_number(num) * last_unit
        elif i > 0 and last_unit and num.isdigit() and i == len(parts) - 1:
            # Spoken shorthand: "3 tỷ 5" = 3.5 tỷ, "3 tỷ 200" = 3.2 tỷ.
            total += float("0." + num) * last_unit
        elif i == 0 and len(parts) == 1:
            n = parse_number(num)
            if n < 100_000:
                raise ValueError(f"ambiguous price {value!r}: add a unit (tỷ, triệu) or give VND")
            total += n
        else:
            raise ValueError(f"cannot read price {value!r}")
    leftover = _PRICE_PART.sub("", raw)
    if re.search(r"[a-z]{2,}", leftover):
        raise ValueError(f"cannot read price {value!r}")
    return Price(round(total), per_m2)


# --- Areas ------------------------------------------------------------------

_DIMS = re.compile(rf"({_NUM})\s*[x×\*]\s*({_NUM})")
_AREA = re.compile(rf"({_NUM})\s*(m2|m²|met vuong|m vuong|sqm|ha|hecta)?")


def parse_area(value: Any) -> float:
    """Square metres from "80m2", "80,5 m²", "5x20", "1 ha"."""
    if isinstance(value, int | float):
        if value <= 0:
            raise ValueError("area must be positive")
        return float(value)
    raw = strip_diacritics(normalize_query(str(value))).lower().strip()
    if not raw:
        raise ValueError("empty area")
    dims = _DIMS.search(raw)
    if dims:
        return round(parse_number(dims.group(1)) * parse_number(dims.group(2)), 2)
    m = _AREA.search(raw)
    if not m:
        raise ValueError(f"cannot read area {value!r}")
    n = parse_number(m.group(1))
    unit = m.group(2) or ""
    if unit in {"ha", "hecta"}:
        n *= 10_000
    leftover = raw[: m.start()] + raw[m.end() :]
    if re.search(r"[a-z]{3,}", leftover.replace("dien tich", "").replace("khoang", "")):
        raise ValueError(f"cannot read area {value!r}")
    return round(n, 2)


# --- Bedrooms and directions -------------------------------------------------

_BEDROOMS = re.compile(r"(\d+)\s*(pn|phong ngu|phong|br|bed|bedrooms?)?\b")


def parse_bedrooms(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = _fold(value)
    if text in {"studio", "1 studio"}:
        return 0
    m = _BEDROOMS.fullmatch(text) or _BEDROOMS.match(text)
    if not m:
        raise ValueError(f"cannot read bedrooms {value!r}")
    return int(m.group(1))


_DIRECTIONS = {
    "dong": "E",
    "tay": "W",
    "nam": "S",
    "bac": "N",
    "dong nam": "SE",
    "dong bac": "NE",
    "tay nam": "SW",
    "tay bac": "NW",
    "dn": "SE",
    "db": "NE",
    "tn": "SW",
    "tb": "NW",
    "east": "E",
    "west": "W",
    "south": "S",
    "north": "N",
    "southeast": "SE",
    "northeast": "NE",
    "southwest": "SW",
    "northwest": "NW",
    "e": "E",
    "w": "W",
    "s": "S",
    "n": "N",
    "se": "SE",
    "ne": "NE",
    "sw": "SW",
    "nw": "NW",
}


def parse_direction(value: Any) -> str:
    text = _fold(value).replace("huong ", "").replace("south east", "southeast")
    text = text.replace("north east", "northeast").replace("south west", "southwest")
    text = text.replace("north west", "northwest")
    if text in _DIRECTIONS:
        return _DIRECTIONS[text]
    raise ValueError(f"cannot read direction {value!r}")


# --- Places -----------------------------------------------------------------


@dataclass(frozen=True)
class District:
    code: str
    city: str
    name: str
    parent: str | None = None


class Gazetteer:
    def __init__(self, data: dict[str, Any]):
        self.districts: dict[str, District] = {}
        self._alias: dict[str, str] = {}
        self._city_alias: dict[str, str] = {}
        for city, cdata in data.get("cities", {}).items():
            for name in cdata.get("names", []):
                self._city_alias[_fold(name)] = city
            for code, entry in (cdata.get("districts") or {}).items():
                if isinstance(entry, dict):
                    aliases, parent = entry.get("aliases", []), entry.get("parent")
                else:
                    aliases, parent = entry, None
                self.add(District(code, city, aliases[0] if aliases else code, parent), aliases)

    def add(self, district: District, aliases: list[str]) -> None:
        self.districts[district.code] = district
        self._alias[_fold(district.code)] = district.code
        for alias in aliases:
            folded = _fold(alias)
            self._alias[folded] = district.code
            for prefix in ("quan ", "huyen ", "tp ", "thanh pho ", "thi xa "):
                if folded.startswith(prefix):
                    self._alias.setdefault(folded[len(prefix) :], district.code)

    def with_aliases(self, extra: dict[str, list[str]]) -> Gazetteer:
        """A copy with tenant-specific aliases, e.g. {"hcm-q7": ["Phú Mỹ Hưng"]}."""
        clone = Gazetteer({})
        clone.districts = dict(self.districts)
        clone._alias = dict(self._alias)
        clone._city_alias = dict(self._city_alias)
        for code, aliases in extra.items():
            if code not in clone.districts:
                raise ValueError(f"unknown district code {code!r}")
            for alias in aliases:
                clone._alias[_fold(alias)] = code
        return clone

    @staticmethod
    def _canonical(text: str) -> str:
        t = _fold(text)
        t = re.sub(r"\b(q|quan|district|dist)\s*(\d{1,2})\b", r"quan \2", t)
        t = re.sub(r"\bh\s+", "huyen ", t)
        t = re.sub(r"\b(tp|thanh pho)\s+", "", t)
        return t

    def district(self, value: Any) -> District:
        text = self._canonical(str(value))
        # Drop a trailing city name: "Quận 7, TP.HCM".
        for alias in sorted(self._city_alias, key=len, reverse=True):
            if text.endswith(" " + alias):
                text = text[: -len(alias) - 1].strip()
                break
        code = self._alias.get(text) or self._alias.get(re.sub(r"^(quan|huyen|thi xa) ", "", text))
        if code is None:
            raise ValueError(f"unknown district {value!r}")
        return self.districts[code]

    def find_in_text(self, text: str) -> list[str]:
        """District codes mentioned in free text, for keyword alias terms."""
        folded = " " + self._canonical(text) + " "
        found: list[str] = []
        for alias, code in self._alias.items():
            if len(alias) < 4 and not alias.startswith("quan"):
                continue  # "q7"-style aliases are already rewritten; skip noisy short ones
            if f" {alias} " in folded and code not in found:
                found.append(code)
        return found

    def descendants(self, code: str) -> list[str]:
        return [code] + [d.code for d in self.districts.values() if d.parent == code]

    def city(self, value: Any) -> str:
        code = self._city_alias.get(_fold(value))
        if code is None:
            raise ValueError(f"unknown city {value!r}")
        return code


@lru_cache(maxsize=1)
def default_gazetteer() -> Gazetteer:
    data = yaml.safe_load((Path(__file__).parent / "gazetteer.yaml").read_text(encoding="utf-8"))
    return Gazetteer(data)


_WARD = re.compile(r"^(phuong|ward|thi tran|xa|tt|p|x)(?:\s+|(?=\d))(.+)$")


def normalize_ward(value: Any) -> str:
    """ "P.5", "Phường 5", "Ward 5" -> "phuong-5"; "P. Thảo Điền" -> "phuong-thao-dien"."""
    text = _fold(value)
    if not text:
        raise ValueError("empty ward")
    m = _WARD.match(text)
    kind, name = ("phuong", text) if not m else (m.group(1), m.group(2))
    kind = {"p": "phuong", "ward": "phuong", "x": "xa", "tt": "thi-tran", "thi tran": "thi-tran"}.get(kind, kind)
    name = re.sub(r"^0+(\d)", r"\1", name.strip())
    return f"{kind}-{name.replace(' ', '-')}"


def slug(value: Any) -> str:
    return _fold(value).replace(" ", "-")
