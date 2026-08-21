"""Small OOXML surgery that python-pptx has no public API for.

Kept tiny and well-commented because raw ``a:``/``p:`` element juggling is the
opposite of self-documenting. Everything else in the builder uses python-pptx's
own API; only these three need the escape hatch:

* table built-in style id (and we default to a *no-style* id so our explicit
  cell fills win),
* paragraph bullet glyph (``a:buChar`` / ``a:buNone``),
* connector arrowheads (``a:headEnd`` / ``a:tailEnd``).

python-pptx *does* expose ``LineFormat.dash_style``, so dashes are handled in the
builder, not here.
"""

from __future__ import annotations

from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

# Built-in PowerPoint table style GUIDs. "No Style, No Grid" gives us a blank
# canvas so our explicit header/band cell fills render exactly as specified
# (the default add_table() style would otherwise paint its own accent banding).
NO_STYLE_NO_GRID = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
NO_STYLE_TABLE_GRID = "{5940675A-B579-460E-94D1-54222C63F5DA}"


def set_table_style(table, guid: str) -> None:
    """Set (or replace) the table's built-in style id."""
    tbl = table._tbl
    tblPr = tbl.find(qn("a:tblPr"))
    if tblPr is None:
        tblPr = tbl.makeelement(qn("a:tblPr"), {})
        tbl.insert(0, tblPr)  # tblPr must be the first child of tbl
    for existing in tblPr.findall(qn("a:tableStyleId")):
        tblPr.remove(existing)
    sid = tblPr.makeelement(qn("a:tableStyleId"), {})
    sid.text = guid
    tblPr.append(sid)


def _clear_bullet_defs(pPr) -> None:
    for tag in ("a:buChar", "a:buAutoNum", "a:buNone", "a:buFont"):
        for existing in pPr.findall(qn(tag)):
            pPr.remove(existing)


def set_bullet_char(paragraph, char: str, *, hang_pt: float = 16) -> None:
    """Force a literal bullet glyph on a paragraph with a hanging indent.

    python-pptx has no bullet API. We drop any inherited bullet defs and add an
    ``a:buChar`` (kept last in ``pPr``, which is schema-valid after ``spcAft``).
    ``marL``/``indent`` give the bullet a proper hang so wrapped lines align.
    """
    pPr = paragraph._p.get_or_add_pPr()
    _clear_bullet_defs(pPr)
    hang = Emu(Pt(hang_pt))
    pPr.set("marL", str(int(hang)))
    pPr.set("indent", str(-int(hang)))
    # NOTE: the char attribute is UNQUALIFIED (`char`, not `a:char`). Namespacing
    # it produces `<a:buChar a:char="…"/>`, which python-pptx/LibreOffice tolerate
    # but real PowerPoint rejects ("PowerPoint could not open the file").
    bu = pPr.makeelement(qn("a:buChar"), {"char": char or "•"})
    pPr.append(bu)


def set_no_bullet(paragraph) -> None:
    """Suppress any inherited bullet on a paragraph."""
    pPr = paragraph._p.get_or_add_pPr()
    _clear_bullet_defs(pPr)
    pPr.append(pPr.makeelement(qn("a:buNone"), {}))


def set_connector_arrows(line_format, arrow: str) -> None:
    """Add arrowheads to a connector's line (``a:headEnd`` / ``a:tailEnd``).

    ``arrow`` is one of none/start/end/both. Appended after any dash/fill on the
    ``a:ln`` element, which is schema-correct (headEnd precedes tailEnd).
    """
    if not arrow or arrow == "none":
        return
    ln = line_format._get_or_add_ln()
    if arrow in ("start", "both"):
        ln.append(ln.makeelement(qn("a:headEnd"), {"type": "triangle"}))
    if arrow in ("end", "both"):
        ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle"}))
