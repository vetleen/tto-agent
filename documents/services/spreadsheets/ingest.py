"""Orchestration for spreadsheet extraction, called from
``process_document._extract_native``: model -> heuristic headers -> mesh ->
vision adjudication (degrading to heuristics) -> chunks + manifest.
"""
from __future__ import annotations

import logging

from documents.services.spreadsheets.chunks import build_spreadsheet_chunks
from documents.services.spreadsheets.headers import detect_headers
from documents.services.spreadsheets.manifest import build_manifest
from documents.services.spreadsheets.mesh import plan_sheet_mesh
from documents.services.spreadsheets.model import load_workbook_model

logger = logging.getLogger(__name__)


def extract_spreadsheet(file_path, version, doc) -> dict:
    """Full spreadsheet ingestion for one version.

    Returns ``{"chunks": [...], "manifest": {...}}`` for the pipeline's
    pre-chunked path. Budget violations raise ``ValueError`` (user-visible
    processing failure); the vision pass can only degrade, never fail this.
    """
    from documents.services.spreadsheets.describe import describe_sheets

    model = load_workbook_model(file_path)
    signal_headers = {s.index: detect_headers(s) for s in model.sheets}
    plans = {
        s.index: plan_sheet_mesh(s, signal_headers[s.index].header_row)
        for s in model.sheets
    }

    try:
        final_headers, descriptions = describe_sheets(model, signal_headers, plans, version, doc)
    except Exception:
        logger.exception(
            "spreadsheets: vision pass crashed for version %s; using heuristic headers", version.id
        )
        final_headers, descriptions = dict(signal_headers), {}

    # A vision override moves the header row, which changes the mesh plan —
    # replan and drop that sheet's already-cached tiles (rendered against the
    # signal-header plan during the vision pass) so geometry stays consistent.
    for sheet in model.sheets:
        final = final_headers[sheet.index]
        if final is not signal_headers[sheet.index]:
            plans[sheet.index] = plan_sheet_mesh(sheet, final.header_row)
            try:
                from documents.services.spreadsheets.tiles import drop_sheet_tiles

                drop_sheet_tiles(version, sheet.index)
            except Exception:
                logger.exception(
                    "spreadsheets: failed to drop stale tiles for sheet %s", sheet.index
                )

    filename = version.native_filename or doc.original_filename or "workbook.xlsx"
    chunks = build_spreadsheet_chunks(model, filename, final_headers, descriptions)
    manifest = build_manifest(model, signal_headers, final_headers, plans, descriptions)
    logger.info(
        "spreadsheets: version_id=%s sheets=%s cells=%s chunks=%s described=%s",
        version.id, len(model.sheets), model.total_cells, len(chunks), len(descriptions),
    )
    return {"chunks": chunks, "manifest": manifest}
