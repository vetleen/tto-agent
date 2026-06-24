"""HTML → PDF rendering for canvas export (WeasyPrint).

The PDF sibling of :mod:`chat.markdown_export` (which does HTML → DOCX). The
markdown→HTML front of the pipeline is shared; this module only turns the
finished HTML body + a stylesheet (``core.styles.build_pdf_css``) into PDF bytes.

WeasyPrint is imported lazily inside :func:`render_canvas_pdf` because its native
Pango libraries aren't present on every dev machine (notably Windows). Import the
module freely; only call the renderer where the libs exist (Linux/Heroku).
"""
from __future__ import annotations

import html as _html
import logging

# WeasyPrint logs '.notdef glyph rendered for Unicode string unsupported by
# fonts: ...' at WARNING once per character that has no glyph in the embedded
# export fonts — e.g. emoji (✅/❌) pasted into a canvas. The PDF still renders
# (the glyph becomes the font's .notdef box), so these are benign, but at
# WARNING they storm Sentry one event per character (WILFRED-66/67). We drop
# just that message off the weasyprint logger; genuinely useful WeasyPrint
# warnings (broken CSS, failed image loads, …) still propagate.
_NOTDEF_PREFIX = ".notdef glyph rendered"


class _DropNotdefGlyphWarnings(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(_NOTDEF_PREFIX)


def _install_weasyprint_log_filter() -> None:
    """Attach the .notdef-dropping filter to the ``weasyprint`` logger once."""
    wp_logger = logging.getLogger("weasyprint")
    if not any(isinstance(f, _DropNotdefGlyphWarnings) for f in wp_logger.filters):
        wp_logger.addFilter(_DropNotdefGlyphWarnings())


def weasyprint_available() -> bool:
    """True when WeasyPrint and its native deps can be imported (skip tests else)."""
    try:
        import weasyprint  # noqa: F401
    except Exception:  # noqa: BLE001  (OSError when Pango is missing, etc.)
        return False
    return True


def _wrap(html_body: str, title: str) -> str:
    # Tag markdown tables (bare ``<table>``) with the class our CSS targets; the
    # inline-styled email-block tables (``<table style=…>``) are left untouched.
    html_body = html_body.replace("<table>", '<table class="wf">')
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        f"<title>{_html.escape(title or 'Document')}</title>"
        f"</head><body>{html_body}</body></html>"
    )


def render_canvas_pdf(html_body: str, *, title: str, css: str) -> bytes:
    """Render a finished HTML body + CSS to PDF bytes.

    CPU-bound and blocking — call under ``sync_to_async``/a thread from the async
    export view so it doesn't stall the event loop. The stylesheet (which carries
    the ``@font-face`` rules) is bound to a ``FontConfiguration`` so embedded
    fonts register reliably.
    """
    from weasyprint import CSS, HTML
    from weasyprint.text.fonts import FontConfiguration

    _install_weasyprint_log_filter()
    font_config = FontConfiguration()
    document = HTML(string=_wrap(html_body, title))
    stylesheet = CSS(string=css, font_config=font_config)
    return document.write_pdf(stylesheets=[stylesheet], font_config=font_config)
