"""Small OOXML surgery that python-pptx has no public API for.

Kept tiny and well-commented because raw ``a:``/``p:``/``c:`` element juggling is
the opposite of self-documenting. Everything else in the builder uses python-pptx's
own API; only these need the escape hatch:

* table built-in style id (and we default to a *no-style* id so our explicit
  cell fills win),
* paragraph bullet glyph (``a:buChar`` / ``a:buNone``),
* connector arrowheads (``a:headEnd`` / ``a:tailEnd``),
* doughnut hole size (``c:holeSize``).

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


def set_bullet_char(paragraph, char: str, *, level: int = 0,
                    hang_pt: float = 18, level_indent_pt: float = 20) -> None:
    """Force a literal bullet glyph on a paragraph with a hanging indent.

    python-pptx has no bullet API. We drop any inherited bullet defs and add an
    ``a:buChar`` (kept last in ``pPr``, which is schema-valid after ``spcAft``).
    ``marL``/``indent`` give the bullet a proper hang so wrapped lines align;
    ``marL`` steps in by ``level_indent_pt`` per nesting level (an explicit marL
    overrides the master's per-level defaults, so without the step every level
    would sit flush left). Both match the Pillow preview's indents.
    """
    pPr = paragraph._p.get_or_add_pPr()
    _clear_bullet_defs(pPr)
    hang = Emu(Pt(hang_pt))
    marl = Emu(Pt(hang_pt + max(0, int(level)) * level_indent_pt))
    pPr.set("marL", str(int(marl)))
    pPr.set("indent", str(-int(hang)))
    # A non-ASCII glyph (‣, ◦, –) references Arial explicitly: the run font may
    # be a face that lacks it (Cambria has no ‣), and buFont must precede buChar
    # in pPr to be schema-valid.
    char = char or "•"
    if any(ord(c) >= 0x80 for c in char):
        pPr.append(pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"}))
    # NOTE: the char attribute is UNQUALIFIED (`char`, not `a:char`). Namespacing
    # it produces `<a:buChar a:char="…"/>`, which python-pptx/LibreOffice tolerate
    # but real PowerPoint rejects ("PowerPoint could not open the file").
    bu = pPr.makeelement(qn("a:buChar"), {"char": char})
    pPr.append(bu)


def set_no_bullet(paragraph) -> None:
    """Suppress any inherited bullet on a paragraph."""
    pPr = paragraph._p.get_or_add_pPr()
    _clear_bullet_defs(pPr)
    pPr.append(pPr.makeelement(qn("a:buNone"), {}))


def set_shape_fill_alpha(shape, opacity: float) -> None:
    """Make a solid-filled shape translucent (``<a:alpha>`` on its fill colour).

    ``opacity`` 0..1. python-pptx has no fill-transparency API. Alpha is in
    thousandths of a percent (100% opaque = 100000).
    """
    val = max(0, min(100000, int(round((opacity or 0) * 100000))))
    spPr = shape._element.spPr
    fill = spPr.find(qn("a:solidFill"))
    if fill is None:
        return
    clr = fill.find(qn("a:srgbClr"))
    if clr is None:
        clr = fill.find(qn("a:schemeClr"))
    if clr is None:
        return
    for ex in clr.findall(qn("a:alpha")):
        clr.remove(ex)
    clr.append(clr.makeelement(qn("a:alpha"), {"val": str(val)}))


def set_shape_outer_shadow(shape, blur_pt: float, dist_pt: float, alpha: float) -> None:
    """A soft black drop shadow straight down (``<a:outerShdw>``) on an autoshape,
    replacing whatever the theme's effect style would have made it inherit.

    ``shape.shadow.inherit = False`` makes python-pptx put an empty
    ``<a:effectLst/>`` in the right spPr slot; the shadow is appended to it.
    """
    shape.shadow.inherit = False
    lst = shape._element.spPr.find(qn("a:effectLst"))
    if lst is None:
        return
    for ex in lst.findall(qn("a:outerShdw")):
        lst.remove(ex)
    shdw = lst.makeelement(qn("a:outerShdw"), {
        "blurRad": str(int(Pt(blur_pt))), "dist": str(int(Pt(dist_pt))),
        "dir": "5400000", "algn": "ctr", "rotWithShape": "0",
    })
    clr = shdw.makeelement(qn("a:srgbClr"), {"val": "000000"})
    clr.append(clr.makeelement(qn("a:alpha"), {"val": str(int(round(alpha * 100000)))}))
    shdw.append(clr)
    lst.append(shdw)


def set_picture_alpha(picture, opacity: float) -> None:
    """Fade a picture (``<a:alphaModFix>`` on its blip). ``opacity`` 0..1."""
    amt = max(0, min(100000, int(round((opacity or 0) * 100000))))
    blip = picture._element.find(qn("p:blipFill"))
    blip = blip.find(qn("a:blip")) if blip is not None else None
    if blip is None:
        return
    for ex in blip.findall(qn("a:alphaModFix")):
        blip.remove(ex)
    blip.append(blip.makeelement(qn("a:alphaModFix"), {"amt": str(amt)}))


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


def set_doughnut_hole(chart, percent: int) -> None:
    """Set a doughnut chart's hole diameter (``c:holeSize``, 10–90 % of the
    overall size). python-pptx has no API for it, so the .pptx would otherwise
    keep the stock ~50 % hole and not match the preview's ``hole`` fraction."""
    percent = max(10, min(90, int(percent)))
    dough = chart._chartSpace.find(qn("c:chart") + "/" + qn("c:plotArea") + "/" + qn("c:doughnutChart"))
    if dough is None:
        return
    hole = dough.find(qn("c:holeSize"))
    if hole is None:
        hole = dough.makeelement(qn("c:holeSize"), {})
        dough.append(hole)
    hole.set("val", str(percent))
