"""Shared file helpers for upload surfaces (documents, meetings, …)."""
from __future__ import annotations

import ntpath
import os


def safe_filename(raw: str, fallback: str = "file", max_length: int = 255) -> str:
    """Normalize and cap a client-provided file name for safe persistence/display.

    Strips any directory components (both POSIX and Windows separators), trims
    whitespace, and caps the length while preserving the extension. Returns
    *fallback* when the input is empty or normalizes away to nothing.
    """
    raw = (raw or "").strip()
    if not raw:
        return fallback
    name = os.path.basename(ntpath.basename(raw)).strip()
    if not name:
        return fallback
    if len(name) <= max_length:
        return name
    base, ext = os.path.splitext(name)
    if not ext:
        return name[:max_length]
    reserved = len(ext)
    if reserved >= max_length:
        return name[:max_length]
    return f"{base[: max_length - reserved]}{ext}"
