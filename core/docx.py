"""Single .docx -> markdown converter shared across data rooms and chat.

The three former converters (documents.services.chunking._load_docx_as_markdown,
chat.services.extract_docx_text, chat.services.import_docx_to_canvas) differed
only in how embedded images were rendered. That behaviour now lives in an
``image_sink`` callback; this module owns the mammoth + markdownify mechanics,
the decompression-bomb guard, and the placeholder/token clean-up.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Callable

# markdownify renders an embedded image as ``![alt](#)``. We put our inline
# text (placeholder or asset token) in ``alt`` and then strip the image syntax
# back down to that text — for both the legacy ``[Image N...]`` placeholders and
# the ``[[image:uuid|...]]`` asset tokens an image_sink may emit.
_ASSET_TOKEN_RE = re.compile(r"!\[(\[\[image:.*?\]\])\]\([^)]*\)")
_PLACEHOLDER_RE = re.compile(r"!\[(\[Image \d+[^\]]*\])\]\([^)]*\)")


def placeholder_image_sink(image, idx: int) -> str:
    """Default sink: a bare ``[Image N]`` placeholder (no description)."""
    return f"[Image {idx}]"


def docx_to_markdown(file, *, image_sink: Callable[[object, int], str]) -> str:
    """Convert a .docx to markdown.

    ``file`` may be a filesystem path, raw ``bytes``, or a binary file-like
    object. ``image_sink(image, idx)`` returns the inline text to place where
    the idx-th embedded image (a mammoth image object) was; it owns any
    description/asset handling. The returned text survives as plain text in the
    markdown.
    """
    import mammoth
    from django.conf import settings
    from markdownify import markdownify as md

    data = _read_bytes(file)

    # Decompression-bomb guard: a small .docx can expand to gigabytes and OOM
    # the worker. Check the declared uncompressed size before mammoth unzips.
    max_uncompressed = getattr(settings, "DOCX_MAX_UNCOMPRESSED_BYTES", 250_000_000)
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            total_uncompressed = sum(info.file_size for info in zf.infolist())
    except zipfile.BadZipFile as exc:
        raise ValueError("This .docx file is corrupt or not a valid Word document.") from exc
    if total_uncompressed > max_uncompressed:
        raise ValueError(
            "This .docx file expands to an unusually large size and can't be processed."
        )

    counter = {"n": 0}

    def convert_image(image):
        counter["n"] += 1
        return {"alt": image_sink(image, counter["n"]), "src": "#"}

    result = mammoth.convert_to_html(
        io.BytesIO(data),
        convert_image=mammoth.images.img_element(convert_image),
    )
    content = md(result.value, heading_style="ATX").strip()
    content = _ASSET_TOKEN_RE.sub(r"\1", content)
    content = _PLACEHOLDER_RE.sub(r"\1", content)
    return content


def _read_bytes(file) -> bytes:
    if isinstance(file, (str, Path)):
        with open(file, "rb") as f:
            return f.read()
    if isinstance(file, (bytes, bytearray)):
        return bytes(file)
    # File-like (UploadedFile, BytesIO, open file). Rewind when possible so a
    # previously-read stream still converts.
    try:
        file.seek(0)
    except (AttributeError, OSError):
        pass
    return file.read()
