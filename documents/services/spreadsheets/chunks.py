"""Chunk emission for spreadsheets: a workbook-overview chunk 0 plus
row-batched, self-contained chunks per sheet.

Chunk text deliberately bypasses ``clean_extracted_text`` — its regexes delete
bare-number lines and ALL-CAPS lines, which are *data* in a spreadsheet (an ID
column, a ``TERMINATED`` status). NUL-stripping already happened in the model.

Only ``text`` is embedded (FTS additionally indexes ``heading`` at weight A),
so every chunk carries its own context prefix: filename, sheet name, the
sheet's vision description, and the row range.
"""
from __future__ import annotations

from core.tokens import count_tokens as _count_tokens
from documents.services.spreadsheets.headers import HeaderInfo, detect_headers
from documents.services.spreadsheets.model import SheetModel, WorkbookModel, col_letter

# Listing more column labels than this in overviews just burns tokens.
_MAX_LISTED_LABELS = 15
_MARKDOWN_MAX_COLS = 40


def _sheet_dims(sheet: SheetModel) -> str:
    cols = len({c for _, cells in sheet.rows for c, _ in cells})
    return f"{cols} columns × {len(sheet.rows):,} rows"


def _row_range(rows: list) -> str:
    first, last = rows[0][0], rows[-1][0]
    return f"Row {first}" if first == last else f"Rows {first}–{last}"


def _row_line(excel_row: int, cells: list, header: HeaderInfo, labelled: bool) -> str:
    if labelled:
        parts = [f"{header.label_for(col)}: {text}" for col, text in cells]
    else:
        parts = [text for _, text in cells]
    return f"Row {excel_row} — " + " | ".join(parts)


def build_spreadsheet_chunks(
    model: WorkbookModel,
    filename: str,
    headers: dict,
    descriptions: dict,
) -> list[dict]:
    """Return the pipeline's chunk dicts for a workbook.

    ``headers``/``descriptions`` map sheet index -> HeaderInfo / description
    text (post vision adjudication when available). Chunk 0 (the overview) is
    always emitted — it is what keeps a chart-only workbook from failing the
    pipeline's "produced 0 chunks" check.
    """
    from django.conf import settings

    target = getattr(settings, "TARGET_CHUNK_TOKENS", 768)
    max_rows = getattr(settings, "XLSX_ROWS_PER_CHUNK_MAX", 25)

    chunks: list[dict] = [_overview_chunk(model, filename, headers, descriptions)]

    for sheet in model.sheets:
        header = headers.get(sheet.index) or HeaderInfo(header_row=None)
        description = (descriptions.get(sheet.index) or "").strip()
        prefix_base = f"{filename}. Sheet '{sheet.name}'"
        if description:
            prefix_base += f": {description.rstrip('.')}"

        batch: list = []
        batch_tokens = 0

        def flush() -> None:
            nonlocal batch, batch_tokens
            if not batch:
                return
            lines = [ln for _, ln in batch]
            text = f"{prefix_base}. {_row_range(batch)}.\n" + "\n".join(lines)
            chunks.append({
                "text": text,
                "token_count": _count_tokens(text),
                "chunk_index": len(chunks),
                "heading": f"Sheet: {sheet.name}",
                "source_page_start": sheet.index,
                "source_page_end": sheet.index,
                "source_offset_start": batch[0][0],
                "source_offset_end": batch[-1][0],
            })
            batch, batch_tokens = [], 0

        for excel_row, cells in sheet.rows:
            if excel_row == header.header_row:
                continue  # labels already live in every data line
            labelled = (
                not header.transposed
                and header.header_row is not None
                and excel_row > header.header_row
            )
            line = _row_line(excel_row, cells, header, labelled)
            line_tokens = _count_tokens(line)
            if batch and (batch_tokens + line_tokens > target or len(batch) >= max_rows):
                flush()
            batch.append((excel_row, line))
            batch_tokens += line_tokens
        flush()

    return chunks


def _overview_chunk(model, filename, headers, descriptions) -> dict:
    lines = [f"Workbook: {filename}"]
    if model.sheets:
        lines.append(f"Sheets ({len(model.sheets)} visible):")
    for sheet in model.sheets:
        header = headers.get(sheet.index) or HeaderInfo(header_row=None)
        if sheet.is_empty:
            lines.append(
                f"{sheet.index + 1}. '{sheet.name}' — no cell data (empty or chart-only)."
            )
            continue
        line = f"{sheet.index + 1}. '{sheet.name}' — {_sheet_dims(sheet)}."
        labels = [header.labels[c] for c in sorted(header.labels)]
        if labels:
            shown = ", ".join(labels[:_MAX_LISTED_LABELS])
            if len(labels) > _MAX_LISTED_LABELS:
                shown += ", …"
            line += f" Columns: {shown}."
        if header.transposed:
            line += " Transposed layout: labels run down the first column."
        if sheet.hidden_cols:
            letters = ", ".join(col_letter(c) for c in sorted(sheet.hidden_cols)[:10])
            line += f" Hidden columns not included: {letters}."
        description = (descriptions.get(sheet.index) or "").strip()
        if description:
            line += f" {description}"
        lines.append(line)
    if model.hidden_sheets:
        lines.append("Hidden sheets (not processed): " + ", ".join(model.hidden_sheets) + ".")
    if model.skipped_sheets:
        lines.append(
            "Additional sheets not processed (over the sheet limit): "
            + ", ".join(model.skipped_sheets) + "."
        )
    text = "\n".join(lines)
    return {
        "text": text,
        "token_count": _count_tokens(text),
        "chunk_index": 0,
        "heading": "Workbook overview",
    }


def workbook_to_markdown(model: WorkbookModel, max_rows_per_sheet: int = 200) -> str:
    """Markdown rendition for email-attachment extraction (``load_documents``).
    Heuristic headers only — this path has no vision pass."""
    parts: list[str] = []
    for sheet in model.sheets:
        parts.append(f"## Sheet: {sheet.name}")
        if sheet.is_empty:
            parts.append("(no cell data)")
            continue
        header = detect_headers(sheet)
        cols = sorted({c for _, cells in sheet.rows for c, _ in cells})
        truncated_cols = len(cols) > _MARKDOWN_MAX_COLS
        cols = cols[:_MARKDOWN_MAX_COLS]

        def cell_text(cells_map, col):
            return cells_map.get(col, "").replace("|", "\\|").replace("\n", " ")

        head = [header.label_for(c).replace("|", "\\|") for c in cols]
        table = ["| " + " | ".join(head) + " |",
                 "| " + " | ".join("---" for _ in cols) + " |"]
        shown = 0
        for excel_row, cells in sheet.rows:
            if excel_row == header.header_row:
                continue
            if shown >= max_rows_per_sheet:
                break
            cells_map = dict(cells)
            table.append("| " + " | ".join(cell_text(cells_map, c) for c in cols) + " |")
            shown += 1
        parts.append("\n".join(table))
        remaining = len(sheet.rows) - shown - (1 if header.header_row else 0)
        if remaining > 0:
            parts.append(f"… ({remaining:,} more rows not shown)")
        if truncated_cols:
            parts.append(f"… (only the first {_MARKDOWN_MAX_COLS} columns shown)")
    if model.hidden_sheets:
        parts.append("Hidden sheets not included: " + ", ".join(model.hidden_sheets))
    return "\n\n".join(parts).strip()
