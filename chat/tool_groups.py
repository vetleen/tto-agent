"""Shared tool groupings used when assembling chat tool lists."""

from __future__ import annotations


# Tools that require at least one attached data room. Keep this as the single
# source of truth for both main-agent and sub-agent context filtering.
DATA_ROOM_TOOL_NAMES = frozenset({
    "document_search",
    "document_read",
    "document_view_image",
    "document_list",
    "document_open_to_canvas",
    "document_edit",
    "document_archive",
    "document_rename",
    "document_version_list",
    "document_version_restore",
    "document_status",
    "canvas_save_to_document",
})
