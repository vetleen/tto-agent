"""Utilities for working with Django file storage in a backend-agnostic way."""

from __future__ import annotations

import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def local_copy(field_file):
    """Yield a local filesystem Path to the contents of a Django FieldFile.

    If the storage backend supports .path (local filesystem), returns the
    existing path directly with no temp file overhead.

    Otherwise (S3, etc.), downloads the file to a temp file, yields the
    temp path, and cleans up on exit.
    """
    # Fast path: local filesystem storage has .path
    try:
        local_path = Path(field_file.path)
        if local_path.exists():
            yield local_path
            return
    except NotImplementedError:
        pass

    # Slow path: download to temp file
    suffix = Path(field_file.name).suffix or ""
    tmp = tempfile.NamedTemporaryFile(
        suffix=suffix, prefix="tto_doc_", delete=False
    )
    try:
        field_file.open("rb")
        try:
            for chunk in field_file.chunks():
                tmp.write(chunk)
        finally:
            field_file.close()
        tmp.close()
        yield Path(tmp.name)
    finally:
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to clean up temp file: %s", tmp.name)
