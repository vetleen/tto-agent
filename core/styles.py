"""Organization "Styles" — design levers for exported documents.

Single source of truth for the small set of typography/colour levers an
organization can configure (Org settings → Styles) and that the canvas → Word
export applies. Kept provider/feature-agnostic so other layouts (e.g. a future
canvas preview) can reuse the same values.

Stored on ``Organization.preferences["styles"]`` (no dedicated model field).
The web settings path uses :func:`get_org_styles`, :func:`validate_styles` and
:data:`FONT_CHOICES`; the export path additionally uses :func:`apply_doc_styles`
(which imports python-docx lazily so the settings page stays light).
"""
from __future__ import annotations

import re

# Zero-width space that chat.services.render_citations injects at the start of
# each numbered source item, so the export can find and shrink just that list
# (a normal numbered list is untouched). Stripped during export.
FOOTNOTE_MARKER = "​"

# Footnote/source list and table text each render this many points below body.
FOOTNOTE_SIZE_DELTA = 2
TABLE_SIZE_DELTA = 2

# Curated, near-universal fonts (ship with Office/Windows/macOS). A ``.docx``
# stores only the font *name*; the reader's app substitutes if it isn't
# installed. The UI also offers a "Custom…" option (sentinel below) for any
# installed font the admin types in.
FONT_CHOICES = [
    {"group": "Sans-serif", "fonts": ["Calibri", "Arial", "Verdana", "Tahoma", "Trebuchet MS", "Segoe UI"]},
    {"group": "Serif", "fonts": ["Cambria", "Times New Roman", "Georgia", "Garamond", "Book Antiqua"]},
]
CUSTOM_FONT_SENTINEL = "__custom__"

# Defaults match today's html2docx output closely (Calibri 11, near-black text)
# so exports for orgs that never touch the setting are essentially unchanged.
# accent_color defaults to "" (empty = opt-in): table headers are only shaded
# once an org sets it.
STYLE_DEFAULTS = {
    "body_font": "Calibri",
    "body_size": 11,
    "heading_font": "Calibri",
    "heading_color": "#1A1A1A",
    "body_color": "#1A1A1A",
    "accent_color": "",
    # Tables. Header rows reuse accent_color for shading; these add borders and
    # optional zebra striping. bold-header, repeat-on-page-break and cell padding
    # are applied unconditionally (sensible defaults, no setting).
    "table_border_style": "all",      # all | horizontal | header | none
    "table_border_color": "#CCCCCC",
    "table_banded": False,
    # Page header & footer (independent typography each). Empty text means "no
    # text"; the page number ("3 / 12") sits bottom-right of the footer.
    "header_text": "",
    "header_font": "Calibri",
    "header_size": 9,
    "header_color": "#1A1A1A",
    "header_bold": False,
    "header_italic": False,
    "footer_text": "",
    "footer_font": "Calibri",
    "footer_size": 9,
    "footer_color": "#1A1A1A",
    "footer_bold": False,
    "footer_italic": False,
    "footer_page_numbers": True,
}

TABLE_BORDER_STYLES = ("all", "horizontal", "header", "none")
HEADER_FOOTER_TEXT_MAX = 200

_FONT_RE = re.compile(r"^[A-Za-z0-9 \-]+$")  # curated + custom like "PT Sans"
_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def get_org_styles(org) -> dict:
    """Resolve an org's effective styles (defaults overlaid with stored values).

    ``org`` may be ``None`` (no membership) → defaults. Pure dict read, no DB,
    so it's safe to call from the async export view. Stored values are written
    through :func:`validate_styles`, so they're trusted here.
    """
    styles = dict(STYLE_DEFAULTS)
    stored = (getattr(org, "preferences", None) or {}).get("styles") if org is not None else None
    if isinstance(stored, dict):
        for key in STYLE_DEFAULTS:
            if key in stored and stored[key] is not None:
                styles[key] = stored[key]
    return styles


def validate_styles(data) -> tuple[dict | None, str | None]:
    """Validate a styles payload from the settings endpoint.

    Returns ``(clean_dict, None)`` on success or ``(None, error_message)``.
    Missing keys fall back to :data:`STYLE_DEFAULTS`.
    """
    if not isinstance(data, dict):
        return None, "Invalid styles payload."

    clean = dict(STYLE_DEFAULTS)

    for key in ("body_font", "heading_font"):
        val = data.get(key, clean[key])
        if not isinstance(val, str):
            return None, f"{key.replace('_', ' ').title()} must be text."
        val = val.strip()
        if not val or len(val) > 64 or not _FONT_RE.match(val):
            return None, "Font names may use letters, numbers, spaces and hyphens (max 64 chars)."
        clean[key] = val

    size = data.get("body_size", clean["body_size"])
    try:
        size = int(size)
    except (TypeError, ValueError):
        return None, "Body size must be a number."
    if not (8 <= size <= 24):
        return None, "Body size must be between 8 and 24."
    clean["body_size"] = size

    for key in ("heading_color", "body_color"):
        val = data.get(key, clean[key])
        if not isinstance(val, str) or not _HEX_RE.match(val.strip()):
            return None, f"{key.replace('_', ' ').title()} must be a hex colour like #1A1A1A."
        clean[key] = val.strip().upper()

    # Accent is optional: "" means "no table-header shading".
    accent = data.get("accent_color", clean["accent_color"])
    if not isinstance(accent, str):
        return None, "Accent colour must be text."
    accent = accent.strip()
    if accent and not _HEX_RE.match(accent):
        return None, "Accent colour must be a hex colour like #2563EB, or left blank."
    clean["accent_color"] = accent.upper() if accent else ""

    border_style = data.get("table_border_style", clean["table_border_style"])
    if border_style not in TABLE_BORDER_STYLES:
        return None, "Invalid table border style."
    clean["table_border_style"] = border_style

    bcol = data.get("table_border_color", clean["table_border_color"])
    if not isinstance(bcol, str) or not _HEX_RE.match(bcol.strip()):
        return None, "Table border colour must be a hex colour like #CCCCCC."
    clean["table_border_color"] = bcol.strip().upper()

    banded = data.get("table_banded", clean["table_banded"])
    if not isinstance(banded, bool):
        return None, "Banded rows must be true or false."
    clean["table_banded"] = banded

    for prefix in ("header", "footer"):
        text = data.get(f"{prefix}_text", clean[f"{prefix}_text"])
        if not isinstance(text, str):
            return None, f"{prefix.title()} text must be text."
        text = text.strip()
        if len(text) > HEADER_FOOTER_TEXT_MAX:
            return None, f"{prefix.title()} text must be {HEADER_FOOTER_TEXT_MAX} characters or fewer."
        clean[f"{prefix}_text"] = text

        font = data.get(f"{prefix}_font", clean[f"{prefix}_font"])
        font = font.strip() if isinstance(font, str) else ""
        if not font or len(font) > 64 or not _FONT_RE.match(font):
            return None, f"{prefix.title()} font names may use letters, numbers, spaces and hyphens (max 64 chars)."
        clean[f"{prefix}_font"] = font

        size = data.get(f"{prefix}_size", clean[f"{prefix}_size"])
        try:
            size = int(size)
        except (TypeError, ValueError):
            return None, f"{prefix.title()} size must be a number."
        if not (6 <= size <= 24):
            return None, f"{prefix.title()} size must be between 6 and 24."
        clean[f"{prefix}_size"] = size

        color = data.get(f"{prefix}_color", clean[f"{prefix}_color"])
        if not isinstance(color, str) or not _HEX_RE.match(color.strip()):
            return None, f"{prefix.title()} colour must be a hex colour like #1A1A1A."
        clean[f"{prefix}_color"] = color.strip().upper()

        for suffix in ("bold", "italic"):
            val = data.get(f"{prefix}_{suffix}", clean[f"{prefix}_{suffix}"])
            if not isinstance(val, bool):
                return None, f"{prefix.title()} {suffix} must be true or false."
            clean[f"{prefix}_{suffix}"] = val

    page_numbers = data.get("footer_page_numbers", clean["footer_page_numbers"])
    if not isinstance(page_numbers, bool):
        return None, "Page numbers must be true or false."
    clean["footer_page_numbers"] = page_numbers

    return clean, None


def _norm_hex(value) -> str | None:
    """Return a 6-char uppercase hex string (no ``#``) or ``None``."""
    if not isinstance(value, str):
        return None
    v = value.strip().lstrip("#")
    if len(v) == 6 and all(c in "0123456789abcdefABCDEF" for c in v):
        return v.upper()
    return None


def _readable_text(hex6: str) -> str:
    """Pick black or white text for a given background (perceived luminance)."""
    r, g, b = int(hex6[0:2], 16), int(hex6[2:4], 16), int(hex6[4:6], 16)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "1A1A1A" if luminance > 150 else "FFFFFF"


def _set_style_font(style, font_name: str) -> None:
    """Force a paragraph style's font name to ``font_name``.

    Word's built-in styles reference a *theme* font (e.g.
    ``w:rFonts w:asciiTheme="majorHAnsi"``), which Word prefers over an explicit
    ``w:ascii`` when both are present — so simply setting ``style.font.name``
    is silently ignored for headings. We set the explicit Latin font names and
    drop the theme references so the chosen font actually renders. Complex-script
    and East-Asian theme refs are left intact (they govern non-Latin scripts).
    """
    from docx.oxml.ns import qn

    rfonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    rfonts.set(qn("w:ascii"), font_name)
    rfonts.set(qn("w:hAnsi"), font_name)
    for theme_attr in ("w:asciiTheme", "w:hAnsiTheme"):
        rfonts.attrib.pop(qn(theme_attr), None)


def _tint(hex6: str, fraction: float) -> str:
    """Blend a colour toward white by ``fraction`` (0 = colour, 1 = white)."""
    r, g, b = int(hex6[0:2], 16), int(hex6[2:4], 16), int(hex6[4:6], 16)
    r = int(r + (255 - r) * fraction)
    g = int(g + (255 - g) * fraction)
    b = int(b + (255 - b) * fraction)
    return f"{r:02X}{g:02X}{b:02X}"


def _style_tables(doc, styles: dict) -> None:
    """Apply border/shading/banding to every table in the document.

    Header rows (row 0) get bold text, optional accent shading (with auto
    contrast text), and repeat on page breaks. Borders, cell padding and
    optional zebra striping follow the org settings. OOXML elements are inserted
    in schema order so Word doesn't flag the file for repair.
    """
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt, RGBColor

    accent = _norm_hex(styles.get("accent_color"))
    border_style = styles.get("table_border_style") or "all"
    border_color = _norm_hex(styles.get("table_border_color")) or "CCCCCC"
    banded = bool(styles.get("table_banded"))
    band_hex = _tint(accent, 0.85) if accent else "F2F2F2"
    try:
        body_size = int(styles.get("body_size"))
    except (TypeError, ValueError):
        body_size = STYLE_DEFAULTS["body_size"]
    table_size = Pt(max(1, body_size - TABLE_SIZE_DELTA))

    def make(tag, **attrs):
        el = OxmlElement(f"w:{tag}")
        for key, value in attrs.items():
            el.set(qn(f"w:{key}"), value)
        return el

    # Successor tags (schema order) used to position inserted elements.
    # insert_element_before() qualifies these names itself — pass them raw.
    _TCPR_AFTER_BORDERS = ("w:shd", "w:noWrap", "w:tcMar", "w:textDirection",
                           "w:tcFitText", "w:vAlign", "w:hideMark")
    _TCPR_AFTER_SHD = _TCPR_AFTER_BORDERS[1:]

    def shade(cell, fill):
        tcPr = cell._tc.get_or_add_tcPr()
        old = tcPr.find(qn("w:shd"))
        if old is not None:
            tcPr.remove(old)
        tcPr.insert_element_before(make("shd", val="clear", color="auto", fill=fill), *_TCPR_AFTER_SHD)

    def header_rule(cell):
        tcPr = cell._tc.get_or_add_tcPr()
        borders = tcPr.find(qn("w:tcBorders"))
        if borders is None:
            borders = make("tcBorders")
            tcPr.insert_element_before(borders, *_TCPR_AFTER_BORDERS)
        old = borders.find(qn("w:bottom"))
        if old is not None:
            borders.remove(old)
        borders.append(make("bottom", val="single", sz="8", space="0", color=border_color))

    def table_borders(table):
        tblPr = table._tbl.tblPr
        old = tblPr.find(qn("w:tblBorders"))
        if old is not None:
            tblPr.remove(old)
        if border_style not in ("all", "horizontal"):
            return
        edges = (["top", "left", "bottom", "right", "insideH", "insideV"]
                 if border_style == "all" else ["top", "bottom", "insideH"])
        borders = make("tblBorders")
        for edge in edges:
            borders.append(make(edge, val="single", sz="4", space="0", color=border_color))
        tblPr.insert_element_before(borders, "w:shd", "w:tblLayout",
                                    "w:tblCellMar", "w:tblLook")

    def cell_margins(table):
        tblPr = table._tbl.tblPr
        old = tblPr.find(qn("w:tblCellMar"))
        if old is not None:
            tblPr.remove(old)
        mar = make("tblCellMar")
        for side in ("top", "left", "bottom", "right"):
            mar.append(make(side, w="80", type="dxa"))  # 80 dxa ≈ 0.06"
        tblPr.insert_element_before(mar, "w:tblLook")

    def repeat_header(row):
        trPr = row._tr.get_or_add_trPr()
        if trPr.find(qn("w:tblHeader")) is None:
            trPr.append(make("tblHeader", val="true"))

    for table in doc.tables:
        table_borders(table)
        cell_margins(table)
        if not table.rows:
            continue

        # Table text renders a couple of points below body, like the sources list.
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.font.size = table_size

        header = table.rows[0]
        repeat_header(header)
        header_text_hex = _readable_text(accent) if accent else None
        for cell in header.cells:
            if accent:
                shade(cell, accent)
            for para in cell.paragraphs:
                for run in para.runs:
                    run.font.bold = True
                    if header_text_hex:
                        run.font.color.rgb = RGBColor.from_string(header_text_hex)
        if border_style == "header":
            for cell in header.cells:
                header_rule(cell)

        if banded:
            for body_index, row in enumerate(table.rows[1:]):
                if body_index % 2 == 1:  # 2nd, 4th, … body row
                    for cell in row.cells:
                        shade(cell, band_hex)


def _resize_footnote_sources(doc, size_pt: int) -> None:
    """Shrink the footnote/source list to ``size_pt``.

    Targets only paragraphs whose first run begins with :data:`FOOTNOTE_MARKER`
    (inserted by ``render_citations``). Strips the marker, sizes every run, and
    sets the paragraph-mark run properties so the auto-generated list number
    matches the text.
    """
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    half_points = str(int(size_pt * 2))
    for para in doc.paragraphs:
        if not para.runs or not para.runs[0].text.startswith(FOOTNOTE_MARKER):
            continue
        para.runs[0].text = para.runs[0].text.replace(FOOTNOTE_MARKER, "", 1)
        for run in para.runs:
            run.font.size = Pt(size_pt)
        pPr = para._p.get_or_add_pPr()
        rPr = pPr.find(qn("w:rPr"))
        if rPr is None:
            rPr = OxmlElement("w:rPr")
            pPr.append(rPr)  # mark rPr is last in the pPr sequence
        old = rPr.find(qn("w:sz"))
        if old is not None:
            rPr.remove(old)
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), half_points)
        rPr.append(sz)


def _add_page_field(paragraph, instr: str) -> None:
    """Append a simple field (e.g. ``PAGE`` / ``NUMPAGES``) to a paragraph.

    python-docx has no field API, so we drop in ``<w:fldSimple>`` with a cached
    "1" the reader recomputes. Inherits the paragraph's (Footer) style.
    """
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), f" {instr} ")
    run = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.text = "1"
    run.append(text)
    fld.append(run)
    paragraph._p.append(fld)


def _style_hf(doc, style_name: str, styles: dict, prefix: str) -> None:
    """Apply one of the ``Header``/``Footer`` styles from ``{prefix}_*`` keys."""
    from docx.shared import Pt, RGBColor

    try:
        style = doc.styles[style_name]
    except KeyError:
        return
    _set_style_font(style, styles.get(f"{prefix}_font") or STYLE_DEFAULTS[f"{prefix}_font"])
    try:
        style.font.size = Pt(int(styles.get(f"{prefix}_size")))
    except (TypeError, ValueError):
        style.font.size = Pt(STYLE_DEFAULTS[f"{prefix}_size"])
    color = _norm_hex(styles.get(f"{prefix}_color"))
    if color:
        style.font.color.rgb = RGBColor.from_string(color)
    style.font.bold = bool(styles.get(f"{prefix}_bold"))
    style.font.italic = bool(styles.get(f"{prefix}_italic"))


def _apply_header_footer(doc, styles: dict) -> None:
    """Add the org's page header/footer with independent typography.

    Header and footer each take their own font/size/colour/bold/italic, applied
    via the built-in ``Header``/``Footer`` styles so every run — including the
    page-number field — inherits the formatting. The footer lays out
    ``footer_text`` on the left and a ``3 / 12`` page indicator on the right via
    a right-aligned tab stop.
    """
    from docx.enum.text import WD_TAB_ALIGNMENT

    header_text = (styles.get("header_text") or "").strip()
    footer_text = (styles.get("footer_text") or "").strip()
    page_numbers = bool(styles.get("footer_page_numbers"))
    if not header_text and not footer_text and not page_numbers:
        return

    if header_text:
        _style_hf(doc, "Header", styles, "header")
    if footer_text or page_numbers:
        _style_hf(doc, "Footer", styles, "footer")

    section = doc.sections[0]

    if header_text:
        section.header.is_linked_to_previous = False
        section.header.paragraphs[0].text = header_text

    if footer_text or page_numbers:
        footer = section.footer
        footer.is_linked_to_previous = False
        para = footer.paragraphs[0]
        para.text = ""
        if page_numbers:
            # The built-in Footer style ships with center + right tabs sized for
            # 1" margins, so a single \t lands the page number at center. Replace
            # them with one right tab at this doc's content width so it sits flush
            # right regardless of margins.
            content_width = section.page_width - section.left_margin - section.right_margin
            try:
                tab_stops = doc.styles["Footer"].paragraph_format.tab_stops
                tab_stops.clear_all()
                tab_stops.add_tab_stop(content_width, WD_TAB_ALIGNMENT.RIGHT)
            except KeyError:
                para.paragraph_format.tab_stops.add_tab_stop(content_width, WD_TAB_ALIGNMENT.RIGHT)
        if footer_text:
            para.add_run(footer_text)
        if page_numbers:
            para.add_run("\t")
            _add_page_field(para, "PAGE")
            para.add_run(" / ")
            _add_page_field(para, "NUMPAGES")


def apply_doc_styles(doc, styles: dict) -> None:
    """Apply resolved org styles to a python-docx ``Document`` in place.

    Sets the ``Normal`` style (body font/size/colour) and ``Heading 1``–``6``
    (font + colour), styles tables (borders, header shading/bold, optional
    banding), shrinks the footnote/source list, and adds the page header/footer.
    Tolerant of missing styles and malformed values (falls back to defaults).
    """
    from docx.shared import Pt, RGBColor

    body_font = styles.get("body_font") or STYLE_DEFAULTS["body_font"]
    heading_font = styles.get("heading_font") or STYLE_DEFAULTS["heading_font"]
    body_color = _norm_hex(styles.get("body_color"))
    heading_color = _norm_hex(styles.get("heading_color"))
    try:
        body_size = int(styles.get("body_size"))
    except (TypeError, ValueError):
        body_size = STYLE_DEFAULTS["body_size"]

    try:
        normal = doc.styles["Normal"]
        _set_style_font(normal, body_font)
        normal.font.size = Pt(body_size)
        if body_color:
            normal.font.color.rgb = RGBColor.from_string(body_color)
    except KeyError:
        pass

    for level in range(1, 7):
        try:
            heading = doc.styles[f"Heading {level}"]
        except KeyError:
            continue
        _set_style_font(heading, heading_font)
        if heading_color:
            heading.font.color.rgb = RGBColor.from_string(heading_color)

    _style_tables(doc, styles)
    _resize_footnote_sources(doc, max(1, body_size - FOOTNOTE_SIZE_DELTA))
    _apply_header_footer(doc, styles)
