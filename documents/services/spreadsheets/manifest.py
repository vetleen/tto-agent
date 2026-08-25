"""Build and read the per-version spreadsheet manifest.

Stored in ``DataRoomDocumentVersion.processing_metadata``. It exists because
(a) the header adjudication and sheet descriptions come from a *paid* vision
call and must never be recomputed, and (b) the tools answer "which tile shows
row 1500" from the mesh geometry without reopening the workbook. The mesh
itself is deterministic from the (immutable) native bytes plus
``RENDERER_REV``, so tiles can be re-rendered lazily and still line up.
"""
from __future__ import annotations

from bisect import bisect_right

from documents.services.spreadsheets.headers import HeaderInfo
from documents.services.spreadsheets.mesh import RENDERER_REV, MeshPlan
from documents.services.spreadsheets.model import WorkbookModel, col_letter

MANIFEST_FORMAT = 1
MANIFEST_TYPE = "spreadsheet"


def _ranges(numbers) -> list:
    """Sorted ints -> [[start, end], ...] runs (compact hidden-row storage)."""
    out: list = []
    for n in sorted(numbers):
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return out


def _mesh_entry(plan: MeshPlan | None) -> dict | None:
    if plan is None:
        return None
    return {
        "orientation": plan.orientation,
        "tile_w": plan.tile_w,
        "tile_h": plan.tile_h,
        "header_row": plan.header_row,
        "bands": [
            {
                "cols": [col_letter(c) for c in band.cols],
                "col_px": list(band.col_px),
                "header_px": band.header_px,
                "row_band_starts": list(band.row_band_starts),
            }
            for band in plan.bands
        ],
    }


def build_manifest(
    model: WorkbookModel,
    signal_headers: dict,
    final_headers: dict,
    plans: dict,
    descriptions: dict,
) -> dict:
    """``signal_headers``/``final_headers`` map sheet index -> HeaderInfo
    (identical objects when no vision adjudication ran); ``plans`` maps sheet
    index -> MeshPlan|None; ``descriptions`` maps sheet index -> str."""
    from django.utils import timezone

    sheets = []
    for sheet in model.sheets:
        signal = signal_headers.get(sheet.index) or HeaderInfo(header_row=None)
        final = final_headers.get(sheet.index) or signal
        sheets.append({
            "index": sheet.index,
            "name": sheet.name,
            "rows": len(sheet.rows),
            "cols": len({c for _, cells in sheet.rows for c, _ in cells}),
            "max_row": sheet.max_data_row,
            "max_col": sheet.max_data_col,
            "empty": sheet.is_empty,
            "header": {
                "row": final.header_row,
                "signal_row": signal.header_row,
                "signal": signal.signal,
                "confidence": signal.confidence,
                "transposed": final.transposed,
                "vision_adjusted": final is not signal,
                "labels": {str(col): label for col, label in sorted(final.labels.items())},
            },
            "description": (descriptions.get(sheet.index) or "").strip(),
            "hidden_cols": [col_letter(c) for c in sorted(sheet.hidden_cols)],
            "hidden_rows": _ranges(sheet.hidden_rows),
            "mesh": _mesh_entry(plans.get(sheet.index)),
        })

    return {
        "format": MANIFEST_FORMAT,
        "type": MANIFEST_TYPE,
        "renderer_rev": RENDERER_REV,
        "generated_at": timezone.now().isoformat(),
        "total_cells": model.total_cells,
        "sheets": sheets,
        "skipped_sheets": list(model.skipped_sheets),
        "hidden_sheets": list(model.hidden_sheets),
    }


# ---------------------------------------------------------------------------
# Read helpers (operate on the stored dict; tolerant of missing pieces)
# ---------------------------------------------------------------------------

def is_spreadsheet_manifest(metadata) -> bool:
    return bool(metadata) and metadata.get("type") == MANIFEST_TYPE


def get_sheet_entry(manifest: dict, sheet) -> dict | None:
    """Resolve a sheet by name (case-insensitive) or index (int or digits)."""
    sheets = (manifest or {}).get("sheets") or []
    if sheet is None:
        return sheets[0] if sheets else None
    if isinstance(sheet, int) or (isinstance(sheet, str) and sheet.strip().isdigit()):
        idx = int(sheet)
        for entry in sheets:
            if entry.get("index") == idx:
                return entry
        return None
    wanted = str(sheet).strip().casefold()
    for entry in sheets:
        if entry.get("name", "").casefold() == wanted:
            return entry
    return None


def tile_y_for_row(band: dict, excel_row: int) -> int | None:
    """y-index of the tile showing ``excel_row`` in one manifest band entry."""
    starts = band.get("row_band_starts") or []
    if not starts or excel_row < starts[0]:
        return None
    return bisect_right(starts, excel_row) - 1


def tiles_covering_rows(sheet_entry: dict, row_start: int, row_end: int) -> list:
    """[(x, y), ...] tiles covering an inclusive row range, across all column
    bands, in (x, y) order."""
    mesh = sheet_entry.get("mesh") or {}
    out = []
    for x, band in enumerate(mesh.get("bands") or []):
        starts = band.get("row_band_starts") or []
        if not starts:
            continue
        y1 = tile_y_for_row(band, row_start)
        y2 = tile_y_for_row(band, row_end)
        if y1 is None:
            y1 = 0
        if y2 is None:
            y2 = 0 if row_end >= starts[0] else -1
        for y in range(y1, y2 + 1):
            out.append((x, y))
    return out


def hidden_cols_set(sheet_entry: dict) -> set:
    from documents.services.spreadsheets.model import col_index

    return {col_index(letter) for letter in sheet_entry.get("hidden_cols") or []}


def row_is_hidden(sheet_entry: dict, excel_row: int) -> bool:
    return any(lo <= excel_row <= hi for lo, hi in sheet_entry.get("hidden_rows") or [])
