"""Worker-side rendering: ``.pptx`` bytes -> PDF (LibreOffice) -> per-slide PNGs.

Only ever runs on the Celery worker (never the web dyno). The heavy memory is a
LibreOffice **subprocess** whose RSS is fully reclaimed on exit — so the hard
part is process hygiene, not memory:

* spawn-per-conversion (no persistent unoserver listener),
* a unique ``-env:UserInstallation`` profile per call (avoids the shared-profile
  lock that hangs concurrent conversions),
* a hard timeout with a **process-group kill** (``os.killpg``) so a hung
  ``soffice.bin`` can't be orphaned and leak,
* a ``TemporaryDirectory`` for input/profile/output,
* a module ``Semaphore`` bounding concurrent soffice spawns (threads pool ->
  one semaphore covers the dyno).

PDF -> PNG uses **pypdfium2** (PDFium, permissively licensed, cross-platform
wheels, no system deps) rather than a poppler/PyMuPDF dependency.
"""

from __future__ import annotations

import io
import logging
import os
import signal
import subprocess
import tempfile
import threading
from pathlib import Path

from django.conf import settings

logger = logging.getLogger(__name__)


class RenderUnavailable(RuntimeError):
    """LibreOffice/rasterizer unavailable or the render failed terminally."""


_semaphore: threading.BoundedSemaphore | None = None
_semaphore_lock = threading.Lock()


def _get_semaphore() -> threading.BoundedSemaphore:
    global _semaphore
    if _semaphore is None:
        with _semaphore_lock:
            if _semaphore is None:
                n = max(1, int(getattr(settings, "SLIDE_RENDER_CONCURRENCY", 1)))
                _semaphore = threading.BoundedSemaphore(n)
    return _semaphore


def _soffice_bin() -> str:
    return getattr(settings, "SOFFICE_BIN", "soffice")


def pptx_to_pdf(pptx_bytes: bytes, *, timeout: int | None = None) -> bytes:
    """Convert ``.pptx`` bytes to PDF bytes via a hardened LibreOffice subprocess."""
    timeout = timeout or int(getattr(settings, "SLIDE_RENDER_TIMEOUT", 120))
    soffice = _soffice_bin()

    with _get_semaphore():
        with tempfile.TemporaryDirectory(prefix="wf_slides_") as tmp:
            tmp_path = Path(tmp)
            profile = tmp_path / "profile"
            in_path = tmp_path / "deck.pptx"
            out_dir = tmp_path / "out"
            out_dir.mkdir()
            in_path.write_bytes(pptx_bytes)

            cmd = [
                soffice, "--headless", "--norestore", "--nolockcheck",
                "--nodefault", "--nologo", "--invisible",
                f"-env:UserInstallation={profile.as_uri()}",
                "--convert-to", "pdf", "--outdir", str(out_dir), str(in_path),
            ]
            proc = _spawn(cmd)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_process_group(proc)
                raise RenderUnavailable(f"LibreOffice render timed out after {timeout}s.")

            pdfs = sorted(out_dir.glob("*.pdf"))
            if not pdfs:
                raise RenderUnavailable("LibreOffice produced no PDF output.")
            return pdfs[0].read_bytes()


def _spawn(cmd: list[str]) -> subprocess.Popen:
    kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "posix":
        # Own session/process group so a timeout can killpg the whole tree —
        # otherwise soffice's forked soffice.bin child is orphaned and leaks.
        kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(cmd, **kwargs)
    except FileNotFoundError as exc:
        raise RenderUnavailable(f"LibreOffice binary not found: {cmd[0]}") from exc


def _kill_process_group(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:  # Windows dev: best-effort tree kill
            proc.kill()
    except Exception:  # noqa: BLE001
        logger.warning("Failed to kill LibreOffice process group", exc_info=True)
    try:
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        pass


def pdf_to_pngs(pdf_bytes: bytes, *, dpi: int | None = None) -> list[tuple[bytes, int, int]]:
    """Rasterize each PDF page to a PNG. Returns ``[(png_bytes, width, height)]``."""
    dpi = dpi or int(getattr(settings, "SLIDE_PREVIEW_DPI", 120))
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover
        raise RenderUnavailable("pypdfium2 is not installed.") from exc

    scale = dpi / 72.0
    out: list[tuple[bytes, int, int]] = []
    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        for i in range(len(doc)):
            page = doc[i]
            bitmap = page.render(scale=scale)
            pil = bitmap.to_pil().convert("RGB")
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            out.append((buf.getvalue(), pil.width, pil.height))
            bitmap.close()
            page.close()
    finally:
        doc.close()
    return out


def render_pptx(pptx_bytes: bytes, *, dpi: int | None = None, timeout: int | None = None):
    """``.pptx`` -> ``(pdf_bytes, [(png_bytes, w, h), ...])`` in one call."""
    pdf_bytes = pptx_to_pdf(pptx_bytes, timeout=timeout)
    return pdf_bytes, pdf_to_pngs(pdf_bytes, dpi=dpi)


def render_available() -> bool:
    """True when the LibreOffice binary and the rasterizer are usable."""
    import shutil

    if shutil.which(_soffice_bin()) is None:
        return False
    try:
        import pypdfium2  # noqa: F401
    except ImportError:
        return False
    return True
