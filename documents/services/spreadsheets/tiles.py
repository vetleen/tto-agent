"""Tile cache: rendered PNG tiles stored as version-owned ``chat.Asset`` rows.

Construction mirrors ``documents.services.image_assets.image_asset_sink``
(``blob`` is an S3-backed FileField). Tiles CASCADE-delete with the version,
are immutable per version (the native bytes never change), and are keyed by
``description`` = ``xlsx-tile:s<sheet>:x<x>:y<y>:rev<RENDERER_REV>`` with
``alt_text`` = TILE_ALT_MARKER — the marker keeps them out of
``_collect_doc_images`` (``document_view_native`` must not attach random
tiles), and the rev in the key invalidates the cache on renderer changes.

The render unit is the column band: one WeasyPrint pass yields all (or the
first ``y_limit``) of a band's y-tiles.
"""
from __future__ import annotations

import hashlib
import logging

from documents.services.spreadsheets.mesh import RENDERER_REV, MeshPlan, plan_sheet_mesh
from documents.services.spreadsheets.model import SheetModel, load_workbook_model
from documents.services.spreadsheets.render import render_band

logger = logging.getLogger(__name__)

TILE_ALT_MARKER = "xlsx-tile"


def tile_key(sheet_index: int, x: int, y: int) -> str:
    return f"xlsx-tile:s{sheet_index}:x{x}:y{y}:rev{RENDERER_REV}"


def get_cached_tiles(version, sheet_index: int, coords) -> dict:
    """{(x, y): Asset} for the requested coords that already exist."""
    from chat.models import Asset

    keys = {tile_key(sheet_index, x, y): (x, y) for x, y in coords}
    out = {}
    for asset in Asset.objects.filter(
        version=version, alt_text=TILE_ALT_MARKER, description__in=list(keys)
    ):
        out[keys[asset.description]] = asset
    return out


def store_tile(version, doc, sheet_index: int, x: int, y: int, png: bytes,
               width: int, height: int):
    """Persist one tile (idempotent on the tile key)."""
    from django.core.files.base import ContentFile

    from chat.models import Asset

    key = tile_key(sheet_index, x, y)
    existing = Asset.objects.filter(
        version=version, alt_text=TILE_ALT_MARKER, description=key
    ).first()
    if existing:
        return existing
    asset = Asset(
        version=version,
        content_type="image/png",
        size_bytes=len(png),
        width=width,
        height=height,
        sha256=hashlib.sha256(png).hexdigest(),
        description=key,
        alt_text=TILE_ALT_MARKER,
        created_by=doc.uploaded_by,
    )
    asset.blob.save(f"{asset.id}.png", ContentFile(png), save=True)
    return asset


def drop_sheet_tiles(version, sheet_index: int) -> None:
    """Delete a sheet's cached tiles (used when the vision pass moves the
    header row, which changes the mesh plan mid-processing)."""
    from chat.models import Asset

    Asset.objects.filter(
        version=version, alt_text=TILE_ALT_MARKER,
        description__startswith=f"xlsx-tile:s{sheet_index}:",
    ).delete()


def render_and_store_band(
    version, doc, sheet: SheetModel, plan: MeshPlan, band_x: int,
    y_limit: int | None = None,
) -> list:
    """Render a band and persist its tiles.

    Returns ``[((x, y), asset, png_bytes), ...]`` in y order — the fresh bytes
    ride along so the vision pass never re-reads what it just wrote.
    """
    tiles = render_band(sheet, plan, band_x, y_limit=y_limit)
    out = []
    for y, (png, width, height) in enumerate(tiles):
        asset = store_tile(version, doc, sheet.index, band_x, y, png, width, height)
        out.append(((band_x, y), asset, png))
    return out


def load_sheet_for_render(version, doc, sheet_name: str, header_row: int | None):
    """Rebuild one sheet's model + plan from the version's native bytes for a
    chat-time lazy render. ``header_row`` comes from the manifest (the
    adjudicated value), NOT re-detection — the plan must match the stored
    geometry. Returns ``(sheet, plan)`` or ``(None, None)``.
    """
    from documents.services.storage_utils import local_copy

    source_file = version.native_blob if version.native_blob else doc.original_file
    if not source_file:
        return None, None
    with local_copy(source_file) as file_path:
        model = load_workbook_model(file_path, only_sheets={sheet_name})
    for sheet in model.sheets:
        if sheet.name == sheet_name:
            return sheet, plan_sheet_mesh(sheet, header_row)
    return None, None


def get_or_render_tiles(version, doc, sheet_entry: dict, coords) -> tuple[dict, list]:
    """Resolve tiles for one manifest sheet entry, rendering missing bands.

    Returns ``({(x, y): Asset}, errors)``. Never raises for render
    unavailability — the caller reports it.
    """
    sheet_index = sheet_entry.get("index", 0)
    mesh = sheet_entry.get("mesh") or {}
    bands = mesh.get("bands") or []

    valid = []
    errors: list[str] = []
    for x, y in coords:
        if 0 <= x < len(bands) and 0 <= y < len(bands[x].get("row_band_starts") or []):
            valid.append((x, y))
        else:
            errors.append(f"Tile x{x}y{y} does not exist on this sheet.")

    found = get_cached_tiles(version, sheet_index, valid)
    missing_bands = sorted({x for x, y in valid if (x, y) not in found})
    if missing_bands:
        sheet, plan = None, None
        try:
            sheet, plan = load_sheet_for_render(
                version, doc, sheet_entry.get("name", ""), mesh.get("header_row")
            )
        except ValueError as exc:
            errors.append(str(exc))
        if sheet is not None and plan is not None:
            for x in missing_bands:
                if x >= len(plan.bands):
                    continue
                try:
                    for coord, asset, _png in render_and_store_band(version, doc, sheet, plan, x):
                        found.setdefault(coord, asset)
                except RuntimeError as exc:
                    errors.append(str(exc))
                    break
                except Exception:
                    logger.exception(
                        "spreadsheets: band render failed version=%s sheet=%s band=%s",
                        version.id, sheet_index, x,
                    )
                    errors.append(f"Rendering tiles for band x{x} failed.")
        elif not errors:
            errors.append("The workbook file for this version is unavailable.")
    return {c: a for c, a in found.items() if c in set(valid)}, errors
