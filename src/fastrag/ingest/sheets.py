"""Read spreadsheets (xlsx, csv) into header + rows."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from fastrag.ingest.parsers import _decode


@dataclass
class Sheet:
    name: str
    rows: list[list[Any]]


def _clean(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        return v or None
    if isinstance(v, datetime | date):
        return v.isoformat()
    return v


def read_xlsx(data: bytes) -> list[Sheet]:
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    sheets = []
    for ws in wb.worksheets:
        if getattr(ws, "sheet_state", "visible") != "visible":
            continue
        rows = [[_clean(c) for c in row] for row in ws.iter_rows(values_only=True)]
        rows = [r for r in rows if any(c is not None for c in r)]
        if rows:
            sheets.append(Sheet(ws.title, rows))
    wb.close()
    return sheets


def read_csv(data: bytes, name: str = "Sheet1") -> list[Sheet]:
    text = _decode(data)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel  # type: ignore[assignment]
    rows = [[_clean(c) for c in r] for r in csv.reader(io.StringIO(text), dialect)]
    return [Sheet(name, [r for r in rows if any(c is not None for c in r)])]


def read_spreadsheet(data: bytes, filename: str) -> list[Sheet]:
    if filename.lower().endswith(".csv"):
        return read_csv(data)
    return read_xlsx(data)
