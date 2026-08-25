"""Tile-mesh planning: pure arithmetic from Excel widths/heights — no pixels,
no openpyxl, no I/O. The renderer later takes its <col> widths and <tr>
heights *from the plan*, so plan and pixels cannot disagree.

The grid is ragged: columns pack into bands that each fit one tile width, and
every column band gets its own y-banding (text wraps differently per band).
The manifest publishes each band's ordered band-start rows so an agent can
compute "row 1500 -> tile x0y17" without opening anything.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from documents.services.spreadsheets.model import SheetModel

# Tile geometry: the largest image no vision provider downscales (Claude's
# standard tier caps the long edge at 1568px and 1568 visual tokens —
# ceil(1568/28) * ceil(768/28) = 56 * 28 = 1568 exactly).
TILE_LANDSCAPE = (1568, 768)
TILE_PORTRAIT = (768, 1568)

LINE_PX = 22  # planned per-line rhythm (21px text line + 1px collapsed gridline)
# Bottom-of-tile reserve: collapsed table borders cost ~1px beyond the row
# sum, and the page must never spill (a spilled table would split one tile
# into two pages and shift every y index after it).
TILE_SAFETY_PX = 8
MAX_WRAP_LINES = 3
MIN_COL_PX = 40
MAX_COL_PX = 420
SNAP_OVERFLOW = 0.12  # absorb a band overflow below this by scaling down
PX_PER_CHAR = 7  # Calibri-11 max digit width; matches the width formula
CELL_PADDING_PX = 9  # 2x4px padding + 1px gridline, mirrored in the renderer
DEFAULT_COL_WIDTH = 8.43  # Excel default column width -> 64px
PORTRAIT_MIN_ROWS = 30
# Bump to invalidate every cached tile (the key embeds it).
RENDERER_REV = 1


def excel_width_to_px(width: float) -> int:
    """Excel column width units -> CSS px (verified: 10.57->79, 34.71->248,
    8.43->64), clamped to keep any column renderable inside one tile."""
    return max(MIN_COL_PX, min(MAX_COL_PX, round(width * 7) + 5))


def points_to_px(points: float) -> int:
    """Row height points -> CSS px at 96/72 DPI (15pt -> 20px)."""
    return max(2, round(points * 4 / 3))


@dataclass(slots=True)
class ColumnBand:
    cols: list  # excel column indices, ascending
    col_px: list  # effective px per column (post-snap scale)
    header_px: int  # px reserved at the top of every tile in this band
    row_px: dict  # excel row -> px, for this band's wrapping
    row_band_starts: list  # excel row opening each y-band (len == ny)


@dataclass(slots=True)
class MeshPlan:
    orientation: str  # "landscape" | "portrait"
    tile_w: int
    tile_h: int
    header_row: int | None
    bands: list = field(default_factory=list)  # ColumnBand, x order

    @property
    def tile_count(self) -> int:
        return sum(len(b.row_band_starts) for b in self.bands)


def _wrap_lines(text_len: int, col_px: int) -> int:
    content_px = max(col_px - CELL_PADDING_PX, 8)
    return max(1, min(MAX_WRAP_LINES, math.ceil(text_len * PX_PER_CHAR / content_px)))


def _band_columns(col_px_map: dict, cols: list, tile_w: int) -> list:
    """Greedy pack columns into bands of <= tile_w px; never split a column.
    A band that the next column would overflow by < SNAP_OVERFLOW absorbs it
    and scales all its columns down proportionally instead of spilling."""
    bands: list[tuple[list, list]] = []
    cur: list = []
    cur_px = 0
    for col in cols:
        px = col_px_map[col]
        if not cur or cur_px + px <= tile_w:
            cur.append(col)
            cur_px += px
            continue
        if cur_px + px <= tile_w * (1 + SNAP_OVERFLOW):
            cur.append(col)
            total = cur_px + px
            factor = tile_w / total
            scaled = [max(1, int(col_px_map[c] * factor)) for c in cur]
            scaled[scaled.index(max(scaled))] += tile_w - sum(scaled)
            bands.append((cur, scaled))
            cur, cur_px = [], 0
            continue
        bands.append((cur, [col_px_map[c] for c in cur]))
        cur, cur_px = [col], px
    if cur:
        bands.append((cur, [col_px_map[c] for c in cur]))
    return bands


def _row_height_px(sheet: SheetModel, excel_row: int, cells, band_cols: dict) -> int:
    custom = sheet.row_heights.get(excel_row)
    if custom is not None:
        return points_to_px(custom)
    lines = 1
    for col, text in cells:
        px = band_cols.get(col)
        if px is not None and text:
            lines = max(lines, _wrap_lines(len(text), px))
            if lines >= MAX_WRAP_LINES:
                break
    return LINE_PX * lines


def plan_sheet_mesh(sheet: SheetModel, header_row: int | None) -> MeshPlan | None:
    """Plan the tile mesh for one sheet. None when there is nothing to show."""
    if sheet.is_empty:
        return None
    cols = sheet.visible_cols()
    if not cols:
        return None

    col_px_map = {
        c: excel_width_to_px(sheet.col_widths.get(c, DEFAULT_COL_WIDTH)) for c in cols
    }
    data_rows = [(r, cells) for r, cells in sheet.rows if r != header_row]

    total_px = sum(col_px_map.values())
    if total_px <= TILE_PORTRAIT[0] and len(data_rows) >= PORTRAIT_MIN_ROWS:
        orientation, (tile_w, tile_h) = "portrait", TILE_PORTRAIT
    else:
        orientation, (tile_w, tile_h) = "landscape", TILE_LANDSCAPE

    header_cells = sheet.row_cells(header_row) if header_row else []

    bands: list[ColumnBand] = []
    for band_cols, band_px in _band_columns(col_px_map, cols, tile_w):
        band_map = dict(zip(band_cols, band_px))
        header_px = (
            _row_height_px(sheet, header_row, header_cells, band_map) if header_row else 0
        )
        capacity = max(tile_h - header_px - TILE_SAFETY_PX, LINE_PX)

        row_px: dict = {}
        starts: list = []
        cur_px = 0
        for excel_row, cells in data_rows:
            h = _row_height_px(sheet, excel_row, cells, band_map)
            row_px[excel_row] = h
            if not starts or cur_px + h > capacity:
                starts.append(excel_row)
                cur_px = h
            else:
                cur_px += h

        bands.append(ColumnBand(
            cols=band_cols, col_px=band_px, header_px=header_px,
            row_px=row_px, row_band_starts=starts,
        ))

    return MeshPlan(
        orientation=orientation, tile_w=tile_w, tile_h=tile_h,
        header_row=header_row, bands=bands,
    )
