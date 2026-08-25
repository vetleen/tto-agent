"""Render one column band of a sheet to PNG tiles.

``build_band_html`` is a pure function (unit-testable anywhere): one
``<table>`` per y-band with explicit ``<colgroup>`` widths and per-row heights
taken *from the mesh plan*, so plan and pixels cannot disagree. Each cell's
content is wrapped in a fixed-height clipping div — real text can never
stretch a row past its planned height. WeasyPrint does not repeat ``<thead>``
across page breaks, so the header row is re-emitted at the top of every
y-band table instead.

Rasterisation (``render_band``) needs WeasyPrint's native deps (no Pango on
Windows — staging/production only); the PDF -> PNG step reuses
``chat.slides.render.pdf_to_pngs`` at dpi=96, where CSS px map 1:1 to device
px, yielding exact tile_w x tile_h images.

Deliberate v1 simplification: horizontal merges render as ``colspan``;
vertical merges render their anchor text once with empty cells beneath
(no rowspan re-emission across band boundaries).
"""
from __future__ import annotations

import html as html_mod
import logging

from documents.services.spreadsheets.mesh import ColumnBand, MeshPlan
from documents.services.spreadsheets.model import SheetModel

logger = logging.getLogger(__name__)

_FONT_STACK = "'Segoe UI', 'DejaVu Sans', 'Helvetica Neue', Arial, sans-serif"
_GRID = "#d4d4d4"
_HEADER_BG = "#f2f2f2"


def _css(plan: MeshPlan) -> str:
    return f"""
@page {{ size: {plan.tile_w}px {plan.tile_h}px; margin: 0; }}
body {{ margin: 0; font-family: {_FONT_STACK}; font-size: 14px; color: #1a1a1a; }}
table {{ table-layout: fixed; border-collapse: collapse; width: {plan.tile_w}px; }}
table.pb {{ page-break-before: always; }}
td, th {{ padding: 0 4px; border: 1px solid {_GRID}; vertical-align: top;
          text-align: left; font-weight: normal; overflow: hidden; }}
th {{ background: {_HEADER_BG}; font-weight: 600; }}
/* 21px text line + 1px collapsed gridline = the plan's 22px row rhythm. */
td div, th div {{ overflow: hidden; line-height: 21px; word-wrap: break-word; }}
"""


def _cell(tag: str, text: str, height_px: int, colspan: int = 1, fill: str | None = None) -> str:
    span = f' colspan="{colspan}"' if colspan > 1 else ""
    style = f' style="background:{fill}"' if fill else ""
    # The clipping div is the geometry enforcer: planned height minus the 1px
    # collapsed gridline, so div + border land exactly on the plan's row_px.
    return (
        f"<{tag}{span}{style}><div style=\"height:{max(height_px - 1, 1)}px\">"
        f"{html_mod.escape(text)}</div></{tag}>"
    )


def _row_html(
    sheet: SheetModel, band: ColumnBand, excel_row: int, height_px: int, tag: str = "td"
) -> str:
    cells_map = dict(sheet.row_cells(excel_row))
    # Horizontal merges anchored in this row -> colspan over the band's columns.
    spans: dict = {}
    covered: set = set()
    for r1, c1, r2, c2 in sheet.merged:
        if r1 == excel_row and c2 > c1:
            in_band = [c for c in band.cols if c1 <= c <= c2]
            if len(in_band) > 1 and in_band[0] == c1:
                spans[c1] = len(in_band)
                covered.update(in_band[1:])
    parts = [f'<tr style="height:{height_px}px">']
    for col in band.cols:
        if col in covered:
            continue
        fill = sheet.cf_fills.get((excel_row, col))
        parts.append(_cell(tag, cells_map.get(col, ""), height_px, spans.get(col, 1), fill))
    parts.append("</tr>")
    return "".join(parts)


def band_row_groups(sheet: SheetModel, plan: MeshPlan, band_x: int) -> list:
    """The band's data rows grouped per y-band: [[excel_row, ...], ...]."""
    band = plan.bands[band_x]
    starts = band.row_band_starts
    groups: list[list[int]] = [[] for _ in starts]
    boundary = 0
    for excel_row, _cells in sheet.rows:
        if excel_row == plan.header_row:
            continue
        while boundary + 1 < len(starts) and excel_row >= starts[boundary + 1]:
            boundary += 1
        groups[boundary].append(excel_row)
    return groups


def build_band_html(
    sheet: SheetModel, plan: MeshPlan, band_x: int, y_limit: int | None = None
) -> str:
    """Self-contained HTML whose printed pages are exactly this band's tiles,
    in y order. ``y_limit`` renders only the first N tiles (the vision pass
    needs just the top one)."""
    band = plan.bands[band_x]
    groups = band_row_groups(sheet, plan, band_x)
    if y_limit is not None:
        groups = groups[:y_limit]

    colgroup = "<colgroup>" + "".join(
        f'<col style="width:{px}px"/>' for px in band.col_px
    ) + "</colgroup>"
    header_html = (
        _row_html(sheet, band, plan.header_row, band.header_px, tag="th")
        if plan.header_row is not None and band.header_px
        else ""
    )

    tables = []
    for i, group in enumerate(groups):
        rows = [
            _row_html(sheet, band, excel_row, band.row_px.get(excel_row, 22))
            for excel_row in group
        ]
        cls = ' class="pb"' if i else ""
        tables.append(f"<table{cls}>{colgroup}{header_html}{''.join(rows)}</table>")

    return f"<html><head><style>{_css(plan)}</style></head><body>{''.join(tables)}</body></html>"


def render_band(
    sheet: SheetModel, plan: MeshPlan, band_x: int, y_limit: int | None = None
) -> list:
    """Rasterise a band -> ``[(png_bytes, width, height), ...]`` in y order.

    Raises RuntimeError when WeasyPrint's native deps are unavailable — callers
    degrade (the pipeline) or report (the chat tool) instead of failing hard.
    """
    from chat.pdf_export import (
        _install_weasyprint_log_filter,
        _safe_url_fetcher,
        weasyprint_available,
    )

    if not weasyprint_available():
        raise RuntimeError("Spreadsheet tile rendering is unavailable in this environment.")

    from weasyprint import HTML

    from chat.slides.render import pdf_to_pngs

    _install_weasyprint_log_filter()
    html = build_band_html(sheet, plan, band_x, y_limit=y_limit)
    pdf = HTML(string=html, url_fetcher=_safe_url_fetcher).write_pdf()
    tiles = pdf_to_pngs(pdf, dpi=96)

    expected = len(band_row_groups(sheet, plan, band_x))
    if y_limit is not None:
        expected = min(expected, y_limit)
    if len(tiles) != expected:
        logger.warning(
            "spreadsheets: band render page mismatch sheet=%r band=%s got=%s expected=%s",
            sheet.name, band_x, len(tiles), expected,
        )
    return tiles
