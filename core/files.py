"""Shared file helpers for upload surfaces (documents, meetings, …)."""
from __future__ import annotations

import hashlib
import logging
import ntpath
import os

logger = logging.getLogger(__name__)

# Read uploads a megabyte at a time so a 50 MB file is never fully resident.
_HASH_READ_SIZE = 1024 * 1024


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


def sha256_of_bytes(data: bytes) -> str:
    """SHA-256 (hex) of an in-memory payload."""
    return hashlib.sha256(data).hexdigest()


def sha256_of_upload(file_obj) -> str:
    """SHA-256 (hex) of an uploaded file's bytes, read in chunks.

    Rewinds the file afterwards so the caller can still persist it. Returns ""
    when the bytes can't be read — content identity is an optimization (dedupe),
    never a gate, so a hash failure must not block the upload it came from.
    """
    digest = hashlib.sha256()
    try:
        for chunk in file_obj.chunks(_HASH_READ_SIZE):
            digest.update(chunk)
    except Exception:
        logger.warning("sha256_of_upload: could not read upload bytes", exc_info=True)
        return ""
    finally:
        try:
            file_obj.seek(0)
        except Exception:
            pass
    return digest.hexdigest()
