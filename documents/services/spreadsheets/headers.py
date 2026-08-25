"""Layered header detection for one sheet — never a single heuristic.

Signal order (strongest first): a ListObject table (its ``tableColumns`` carry
the authoritative labels) → the autofilter range → a frozen pane (a frozen
*column* with no frozen row signals a transposed sheet: labels down column A)
→ a value-type profile (a mostly-text first row over mostly-non-text data
rows) → positional column letters at low confidence.

The per-sheet vision pass later *adjudicates* the result; the manifest records
both the signal-derived and the final values so the override is auditable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from documents.services.spreadsheets.model import SheetModel, col_letter

# Minimum share of text cells for a type-profile header row, and the maximum
# share for the data rows beneath it.
_HEADER_TEXT_SHARE = 0.7
_DATA_TEXT_SHARE = 0.5
_MIN_HEADER_CELLS = 2
_MAX_LABEL_CHARS = 80

_WS_RE = re.compile(r"\s+")


@dataclass(slots=True)
class HeaderInfo:
    header_row: int | None  # Excel row number of the header, when one exists
    labels: dict = field(default_factory=dict)  # excel col -> label text
    transposed: bool = False  # labels run down the first column instead
    confidence: str = "low"  # definitive | strong | medium | low
    signal: str = "positional"  # table | autofilter | freeze | freeze_col | type_profile | positional | empty

    def label_for(self, col: int) -> str:
        label = self.labels.get(col, "")
        return label or f"Column {col_letter(col)}"


def _clean_label(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()[:_MAX_LABEL_CHARS]


def _labels_from_row(sheet: SheetModel, excel_row: int) -> dict:
    return {
        col: _clean_label(text)
        for col, text in sheet.row_cells(excel_row)
        if text and text.strip()
    }


def detect_headers(sheet: SheetModel) -> HeaderInfo:
    if sheet.is_empty:
        return HeaderInfo(header_row=None, confidence="low", signal="empty")

    # 1. ListObject table — definitive; its column names are the labels.
    tables = [t for t in sheet.tables if t.header_row_count >= 1]
    if tables:
        table = max(
            tables,
            key=lambda t: (t.bounds[2] - t.bounds[0] + 1) * (t.bounds[3] - t.bounds[1] + 1),
        )
        min_row, min_col, _, max_col = table.bounds
        labels = {
            min_col + i: _clean_label(name)
            for i, name in enumerate(table.column_names)
            if name and min_col + i <= max_col
        }
        if not labels:
            labels = _labels_from_row(sheet, min_row)
        return HeaderInfo(header_row=min_row, labels=labels, confidence="definitive", signal="table")

    # 2. Autofilter range — very strong; its first row is the header.
    if sheet.auto_filter:
        row = sheet.auto_filter[0]
        return HeaderInfo(
            header_row=row, labels=_labels_from_row(sheet, row),
            confidence="strong", signal="autofilter",
        )

    # 3. Frozen panes. Frozen rows put the header on the last frozen row; a
    # frozen column with no frozen row marks a transposed sheet.
    if sheet.freeze_rows >= 1:
        row = sheet.freeze_rows
        return HeaderInfo(
            header_row=row, labels=_labels_from_row(sheet, row),
            confidence="strong", signal="freeze",
        )
    if sheet.freeze_cols >= 1:
        return HeaderInfo(header_row=None, transposed=True, confidence="strong", signal="freeze_col")

    # 4. Type profile: a mostly-text row directly above mostly-non-text rows.
    profile = sheet.type_profile
    for i, (excel_row, n_cells, n_text) in enumerate(profile):
        if n_cells < _MIN_HEADER_CELLS or n_text / n_cells < _HEADER_TEXT_SHARE:
            continue
        following = profile[i + 1 : i + 4]
        if following and any(
            n and (t / n) <= _DATA_TEXT_SHARE for _, n, t in following
        ):
            return HeaderInfo(
                header_row=excel_row, labels=_labels_from_row(sheet, excel_row),
                confidence="medium", signal="type_profile",
            )
        break  # only the first text-heavy candidate is considered

    # 5. Positional fallback — chunks label cells by column letter.
    return HeaderInfo(header_row=None, confidence="low", signal="positional")
