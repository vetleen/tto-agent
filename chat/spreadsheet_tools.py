"""Spreadsheet chat tools: exact reads (document_read_sheet) and rendered
tile views (document_view_sheet).

A processed spreadsheet has three surfaces: row chunks for retrieval, these
exact reads for values (a substring scan beats embeddings at row lookup), and
the tile mesh for layout. Both tools are manifest-driven — sheet geometry,
header labels and the row->tile mapping come from
``version.processing_metadata`` — and only ``find``/``rows``/``cell`` reads
reopen the workbook (streaming, read-only).

Gates fail closed and are deliberately STRICTER than document_read: direct
workbook reads and tiles bypass per-chunk quarantine, so a fully OR partially
quarantined version refuses (document_read merely omits the quarantined
chunks).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from pydantic import BaseModel, Field

from llm.tools import ContextAwareTool, ReasonBaseModel, get_tool_registry

logger = logging.getLogger(__name__)

_TILE_RE = re.compile(r"^x(\d+)y(\d+)$")
_ROWS_RE = re.compile(r"^(\d+)(?:\s*[-–]\s*(\d+))?$")


def _resolve_spreadsheet(context, doc_index: int, data_room_id):
    """-> (doc, version, manifest, error) — exactly one side is None."""
    from chat.tools import _resolve_document
    from documents.models import DataRoomDocument
    from documents.services.spreadsheets.manifest import is_spreadsheet_manifest

    doc, err = _resolve_document(context, doc_index, data_room_id)
    if err:
        return None, None, None, err
    # READY only; same "not found" message as the other tools — never leak
    # scan state.
    if doc.status != DataRoomDocument.Status.READY:
        return None, None, None, f"No document with index {doc_index} found."
    version = doc.active_searchable_version or doc.current_version
    if version is None:
        return None, None, None, f"No document with index {doc_index} found."
    if version.is_quarantined or version.is_partially_quarantined:
        return None, None, None, (
            "This document is quarantined (fully or partially), so its "
            "spreadsheet cannot be read or viewed directly."
        )
    manifest = version.processing_metadata
    if not is_spreadsheet_manifest(manifest):
        return None, None, None, (
            "This document is not a spreadsheet, or it was processed before "
            "spreadsheet support existed — re-upload it to enable sheet reads."
        )
    return doc, version, manifest, None


def _native_source(version, doc):
    source = version.native_blob if version.native_blob else doc.original_file
    if not source:
        raise ValueError("The workbook file for this version is unavailable.")
    return source


def _letter_labels(entry: dict) -> dict:
    """Manifest labels ({"2": "Name"}) keyed by column letter ({"B": "Name"})."""
    from documents.services.spreadsheets.model import col_letter

    labels = (entry.get("header") or {}).get("labels") or {}
    return {col_letter(int(k)): v for k, v in labels.items()}


def _label_for(entry: dict, col: int) -> str:
    from documents.services.spreadsheets.model import col_letter

    labels = (entry.get("header") or {}).get("labels") or {}
    return labels.get(str(col)) or f"Column {col_letter(col)}"


def _tile_for_cell(entry: dict, row: int, col: int) -> str | None:
    from documents.services.spreadsheets.manifest import tile_y_for_row
    from documents.services.spreadsheets.model import col_index

    for x, band in enumerate((entry.get("mesh") or {}).get("bands") or []):
        cols = [col_index(letter) for letter in band.get("cols") or []]
        if col in cols:
            y = tile_y_for_row(band, row)
            return f"x{x}y{y}" if y is not None else None
    return None


def _tile_refs(entry: dict, row_start: int, row_end: int, cap: int = 12) -> list:
    from documents.services.spreadsheets.manifest import tiles_covering_rows

    refs = [f"x{x}y{y}" for x, y in tiles_covering_rows(entry, row_start, row_end)]
    return refs[:cap]


def _tile_extent(entry: dict, x: int, y: int) -> str:
    """Human description of one tile's coverage: "columns A–J, rows 2–45"."""
    bands = (entry.get("mesh") or {}).get("bands") or []
    band = bands[x]
    cols = band.get("cols") or []
    starts = band.get("row_band_starts") or []
    r1 = starts[y]
    r2 = (starts[y + 1] - 1) if y + 1 < len(starts) else entry.get("max_row", r1)
    col_part = f"column {cols[0]}" if len(cols) == 1 else f"columns {cols[0]}–{cols[-1]}"
    row_part = f"row {r1}" if r1 == r2 else f"rows {r1}–{r2}"
    return f"{col_part}, {row_part}"


def _sheet_summary(entry: dict) -> dict:
    header = entry.get("header") or {}
    mesh = entry.get("mesh") or {}
    summary = {
        "index": entry.get("index"),
        "name": entry.get("name"),
        "rows": entry.get("rows"),
        "cols": entry.get("cols"),
        "header_row": header.get("row"),
        "columns": _letter_labels(entry),
    }
    if entry.get("empty"):
        summary["empty"] = True
    if header.get("transposed"):
        summary["transposed"] = True
    if entry.get("description"):
        summary["description"] = entry["description"]
    if entry.get("hidden_cols"):
        summary["hidden_cols"] = entry["hidden_cols"]
    if mesh:
        summary["tiles"] = [
            {
                "band": x,
                "cols": f"{(b.get('cols') or ['?'])[0]}–{(b.get('cols') or ['?'])[-1]}",
                "tiles_y": len(b.get("row_band_starts") or []),
            }
            for x, b in enumerate(mesh.get("bands") or [])
        ]
    return summary


class ReadSheetInput(ReasonBaseModel):
    doc_index: int = Field(description="Index of the spreadsheet document in the attached data rooms.")
    sheet: Optional[str] = Field(
        default=None,
        description="Sheet name or index. Defaults to all sheets for the overview and for find, and to the first sheet for cell/rows reads.",
    )
    find: Optional[str] = Field(
        default=None,
        description="Case-insensitive substring to locate across cell values (like Ctrl+F). Returns exact cell locations.",
    )
    cell: Optional[str] = Field(
        default=None, description='Read a single cell by reference, e.g. "B7". Returns its full row for context.'
    )
    rows: Optional[str] = Field(
        default=None, description='Read an exact row range, e.g. "344-353" or "344".'
    )
    cols: Optional[str] = Field(
        default=None, description='Restrict a rows read to these columns, e.g. "A,C-F".'
    )
    data_room_id: Optional[int] = Field(
        default=None, description="Optional data room id to disambiguate the document index."
    )


class DocumentReadSheetTool(ContextAwareTool):
    """Exact, manifest-aware reads from a spreadsheet document."""

    name: str = "document_read_sheet"
    section: str = "skills"
    subagent_section: str = "chat"
    start_label: str = "Reading spreadsheet..."
    end_label: str = "Read spreadsheet"
    description: str = (
        "Read a spreadsheet (.xlsx/.xlsm) document exactly, by document index. "
        "With no extra arguments: a workbook overview (sheets, sizes, column "
        "headers, descriptions, tile grid). find= locates a value anywhere in "
        "the workbook (case-insensitive substring, like Ctrl+F) and returns "
        "exact cells — prefer this over document_search for looking up "
        "specific rows or values. cell=/rows=/cols= read exact cells with "
        "header labels. Results include the tile coordinates covering the "
        "rows, for document_view_sheet."
    )
    args_schema: type[BaseModel] = ReadSheetInput

    def end_label_for_result(self, result: dict) -> str | None:
        if not isinstance(result, dict) or result.get("error"):
            return None
        if "matches" in result:
            n = result.get("match_count", 0)
            return f"Found {n} match{'' if n == 1 else 'es'} in the spreadsheet"
        if "rows" in result:
            return f"Read {len(result['rows'])} row{'' if len(result['rows']) == 1 else 's'} from '{result.get('sheet', '')}'"
        if "cell" in result:
            return f"Read cell {result['cell']}"
        if "sheets" in result:
            n = len(result["sheets"])
            return f"Listed {n} sheet{'' if n == 1 else 's'}"
        return None

    def _run(
        self,
        doc_index: int,
        sheet: str | None = None,
        find: str | None = None,
        cell: str | None = None,
        rows: str | None = None,
        cols: str | None = None,
        data_room_id: int | None = None,
        **kwargs,
    ) -> str:
        doc, version, manifest, err = _resolve_spreadsheet(self.context, doc_index, data_room_id)
        if err:
            return json.dumps({"error": err})
        try:
            if find:
                result = self._do_find(doc, version, manifest, find, sheet)
            elif cell:
                result = self._do_cell(doc, version, manifest, cell, sheet)
            elif rows or cols:
                result = self._do_rows(doc, version, manifest, rows, cols, sheet)
            else:
                result = self._do_overview(doc, manifest, sheet)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        result["doc_index"] = doc_index
        result["file"] = doc.original_filename
        return json.dumps(result, ensure_ascii=False)

    # -- modes ---------------------------------------------------------------

    def _do_overview(self, doc, manifest: dict, sheet: str | None) -> dict:
        from documents.services.spreadsheets.manifest import get_sheet_entry

        if sheet is not None:
            entry = get_sheet_entry(manifest, sheet)
            if entry is None:
                raise ValueError(f"No sheet {sheet!r} in this workbook.")
            entries = [entry]
        else:
            entries = manifest.get("sheets") or []
        result = {"sheets": [_sheet_summary(e) for e in entries]}
        if manifest.get("hidden_sheets"):
            result["hidden_sheets"] = manifest["hidden_sheets"]
        if manifest.get("skipped_sheets"):
            result["unprocessed_sheets"] = manifest["skipped_sheets"]
        result["hint"] = (
            "Use find= to locate values, rows=/cell= for exact reads, and "
            "document_view_sheet with the tile coordinates to see the layout."
        )
        return result

    def _do_find(self, doc, version, manifest: dict, needle: str, sheet: str | None) -> dict:
        from django.conf import settings

        from documents.services.spreadsheets.manifest import (
            get_sheet_entry,
            hidden_cols_set,
            row_is_hidden,
        )
        from documents.services.spreadsheets.model import col_letter
        from documents.services.spreadsheets.reads import find_in_workbook
        from documents.services.storage_utils import local_copy

        sheet_names = None
        if sheet is not None:
            entry = get_sheet_entry(manifest, sheet)
            if entry is None:
                raise ValueError(f"No sheet {sheet!r} in this workbook.")
            sheet_names = {entry["name"]}

        with local_copy(_native_source(version, doc)) as path:
            hits, truncated = find_in_workbook(
                path, needle, sheet_names=sheet_names,
                max_results=getattr(settings, "XLSX_FIND_MAX_RESULTS", 50),
                max_cell_chars=getattr(settings, "XLSX_MAX_CELL_CHARS", 500),
            )

        matches = []
        for hit in hits:
            entry = get_sheet_entry(manifest, hit["sheet"]) or {}
            match = {
                "sheet": hit["sheet"],
                "cell": f"{col_letter(hit['col'])}{hit['row']}",
                "row": hit["row"],
                "label": _label_for(entry, hit["col"]),
                "value": hit["value"],
            }
            if row_is_hidden(entry, hit["row"]) or hit["col"] in hidden_cols_set(entry):
                match["hidden"] = True
            tile = _tile_for_cell(entry, hit["row"], hit["col"])
            if tile:
                match["tile"] = tile
            matches.append(match)
        result = {"query": needle, "match_count": len(matches), "matches": matches}
        if truncated:
            result["truncated"] = True
            result["hint"] = "More matches exist — narrow the query or scope it with sheet=."
        elif matches:
            result["hint"] = "Read the full rows with rows=, or view them with document_view_sheet."
        return result

    def _do_rows(self, doc, version, manifest: dict, rows: str | None,
                 cols: str | None, sheet: str | None) -> dict:
        from django.conf import settings

        from documents.services.spreadsheets.manifest import get_sheet_entry, row_is_hidden
        from documents.services.spreadsheets.model import col_letter
        from documents.services.spreadsheets.reads import parse_col_spec, read_rows
        from documents.services.storage_utils import local_copy

        entry = get_sheet_entry(manifest, sheet)
        if entry is None:
            raise ValueError(
                f"No sheet {sheet!r} in this workbook." if sheet is not None
                else "This workbook has no sheets."
            )
        if rows:
            m = _ROWS_RE.match(rows.strip())
            if not m:
                raise ValueError(f'Invalid rows range {rows!r} — use e.g. "344-353".')
            row_start = int(m.group(1))
            row_end = int(m.group(2) or m.group(1))
            if row_end < row_start:
                row_start, row_end = row_end, row_start
        else:
            row_start, row_end = 1, entry.get("max_row") or 1
        col_list = parse_col_spec(cols) if cols else None

        with local_copy(_native_source(version, doc)) as path:
            data, truncated = read_rows(
                path, entry["name"], row_start, row_end, cols=col_list,
                max_rows=getattr(settings, "XLSX_READ_MAX_ROWS", 200),
                max_cell_chars=getattr(settings, "XLSX_MAX_CELL_CHARS", 500),
            )

        header_row = (entry.get("header") or {}).get("row")
        out_rows = []
        for excel_row, cells in data:
            row_out = {
                "row": excel_row,
                "cells": {col_letter(c): text for c, text in cells},
            }
            if excel_row == header_row:
                row_out["is_header"] = True
            if row_is_hidden(entry, excel_row):
                row_out["hidden"] = True
            out_rows.append(row_out)

        used_cols = sorted({c for _, cells in data for c, _ in cells})
        result = {
            "sheet": entry["name"],
            "range": f"{row_start}-{row_end}",
            "columns": {col_letter(c): _label_for(entry, c) for c in used_cols},
            "rows": out_rows,
        }
        if truncated:
            result["truncated"] = True
            result["hint"] = (
                f"Only the first {len(out_rows)} rows are shown — request a "
                "narrower rows= range for the rest."
            )
        tiles = _tile_refs(entry, row_start, min(row_end, entry.get("max_row") or row_end))
        if tiles:
            result["tiles"] = tiles
        return result

    def _do_cell(self, doc, version, manifest: dict, cell: str, sheet: str | None) -> dict:
        from django.conf import settings

        from documents.services.spreadsheets.manifest import get_sheet_entry
        from documents.services.spreadsheets.model import col_letter, parse_cell_ref
        from documents.services.spreadsheets.reads import read_cell
        from documents.services.storage_utils import local_copy

        entry = get_sheet_entry(manifest, sheet)
        if entry is None:
            raise ValueError(
                f"No sheet {sheet!r} in this workbook." if sheet is not None
                else "This workbook has no sheets."
            )
        row, col = parse_cell_ref(cell)
        with local_copy(_native_source(version, doc)) as path:
            value, row_cells = read_cell(
                path, entry["name"], row, col,
                max_cell_chars=getattr(settings, "XLSX_MAX_CELL_CHARS", 500),
            )
        result = {
            "sheet": entry["name"],
            "cell": f"{col_letter(col)}{row}",
            "label": _label_for(entry, col),
            "value": value,
            "row_values": {f"{_label_for(entry, c)} ({col_letter(c)})": text for c, text in row_cells},
        }
        tile = _tile_for_cell(entry, row, col)
        if tile:
            result["tile"] = tile
        return result


class ViewSheetInput(ReasonBaseModel):
    doc_index: int = Field(description="Index of the spreadsheet document in the attached data rooms.")
    sheet: Optional[str] = Field(
        default=None, description="Sheet name or index. Defaults to the first sheet."
    )
    tiles: Optional[list[str]] = Field(
        default=None,
        description='Tile coordinates to view, e.g. ["x0y0", "x1y2"] — from the tile grid in document_read_sheet results.',
    )
    rows: Optional[str] = Field(
        default=None,
        description='View the tiles covering a row range instead, e.g. "1490-1510".',
    )
    data_room_id: Optional[int] = Field(
        default=None, description="Optional data room id to disambiguate the document index."
    )


class DocumentViewSheetTool(ContextAwareTool):
    """Attach rendered spreadsheet tiles so the model can see the layout."""

    name: str = "document_view_sheet"
    section: str = "skills"
    subagent_section: str = "chat"
    start_label: str = "Viewing sheet..."
    end_label: str = "Viewed sheet"
    description: str = (
        "View a spreadsheet sheet as rendered image tiles (real column widths, "
        "merged cells, highlighting) to understand its LAYOUT. Pass tiles= "
        "coordinates from document_read_sheet, or rows= to view the tiles "
        "covering those rows; with neither, the top tile of each column band "
        "is shown. Use document_read_sheet for exact values — do not read "
        "numbers off the images."
    )
    args_schema: type[BaseModel] = ViewSheetInput

    def _run(
        self,
        doc_index: int,
        sheet: str | None = None,
        tiles: list[str] | None = None,
        rows: str | None = None,
        data_room_id: int | None = None,
        **kwargs,
    ) -> str:
        import base64

        from django.conf import settings

        from chat.assets import image_token
        from documents.services.spreadsheets.manifest import (
            get_sheet_entry,
            tiles_covering_rows,
        )
        from documents.services.spreadsheets.tiles import get_or_render_tiles

        doc, version, manifest, err = _resolve_spreadsheet(self.context, doc_index, data_room_id)
        if err:
            return err
        entry = get_sheet_entry(manifest, sheet)
        if entry is None:
            return f"No sheet {sheet!r} in this workbook." if sheet is not None else "This workbook has no sheets."
        mesh = entry.get("mesh") or {}
        bands = mesh.get("bands") or []
        if entry.get("empty") or not bands:
            return f"Sheet '{entry.get('name')}' has no cell data to display."

        notes: list[str] = []
        if tiles:
            coords = []
            for ref in tiles:
                m = _TILE_RE.match(str(ref).strip())
                if not m:
                    notes.append(f'Invalid tile reference {ref!r} — use e.g. "x0y0".')
                    continue
                coords.append((int(m.group(1)), int(m.group(2))))
        elif rows:
            m = _ROWS_RE.match(rows.strip())
            if not m:
                return f'Invalid rows range {rows!r} — use e.g. "1490-1510".'
            r1 = int(m.group(1))
            r2 = int(m.group(2) or m.group(1))
            coords = tiles_covering_rows(entry, min(r1, r2), max(r1, r2))
            if not coords:
                return f"No tiles cover rows {rows} on sheet '{entry.get('name')}'."
        else:
            coords = [(x, 0) for x in range(len(bands)) if bands[x].get("row_band_starts")]

        coords = list(dict.fromkeys(coords))  # dedupe, order preserved
        cap = getattr(settings, "XLSX_MAX_TILES_PER_VIEW", 6)
        if len(coords) > cap:
            notes.append(
                f"Requested {len(coords)} tiles; showing the first {cap} — "
                "request the rest in another call."
            )
            coords = coords[:cap]
        if not coords:
            return "\n".join(notes) or "No tiles requested."

        assets, errors = get_or_render_tiles(version, doc, entry, coords)
        notes.extend(errors)

        attached = 0
        lines: list[str] = []
        for coord in coords:
            asset = assets.get(coord)
            if asset is None:
                continue
            x, y = coord
            extent = _tile_extent(entry, x, y)
            with asset.blob.open("rb") as fh:
                png = fh.read()
            description = f"Sheet '{entry.get('name')}' tile x{x}y{y} ({extent})"
            token = image_token(asset.id, description)
            self.context.pending_native_assets.append({
                "asset_id": token,
                "b64": base64.b64encode(png).decode("ascii"),
                "media_type": "image/png",
                "description": description,
            })
            attached += 1
            lines.append(
                f"Attached tile x{x}y{y} ({extent}). To show it to the user, "
                f"paste: {token}"
            )
        lines.extend(notes)
        if attached == 0:
            return "\n".join(lines) or "No tiles could be attached."
        lines.append(
            "(The tile image(s) are now visible to you below. They show layout — "
            "read exact values with document_read_sheet.)"
        )
        return "\n".join(lines)

    def end_label_for_result(self, result: dict) -> str | None:
        return None  # prose result; static end_label applies


_registry = get_tool_registry()
_registry.register_tool(DocumentReadSheetTool())
_registry.register_tool(DocumentViewSheetTool())
