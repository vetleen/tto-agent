"""Generate the branded base template ``wilfred_default.pptx``.

The template carries the real PowerPoint theme: its ``a:clrScheme`` (the 12
native colour slots) and ``a:fontScheme`` (major/minor font) come from
:data:`chat.slides.theme.WILFRED_BASE_THEME`. Because the builder references
native slots via ``MSO_THEME_COLOR`` (``schemeClr``), decks then render in brand
colours AND those colours appear in PowerPoint's colour picker for downstream
manual editing.

We start from python-pptx's stock ``default.pptx`` and rewrite only
``ppt/theme/theme1.xml`` in the zip — the stock layouts/master are left intact
(the builder only ever uses the Blank layout, and unused layouts are cheap).

The output is deterministic package data committed to the repo; regenerate with
``manage.py build_slide_template`` after changing the theme.
"""

from __future__ import annotations

import io
import os
import zipfile

from lxml import etree

from chat.slides.theme import WILFRED_BASE_THEME

_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_NS = {"a": _A}

# clrScheme child tag (localname) -> theme colour key. Same names, but explicit
# so a future divergence is obvious.
_SLOT_KEYS = (
    "dk1", "lt1", "dk2", "lt2",
    "accent1", "accent2", "accent3", "accent4", "accent5", "accent6",
    "hlink", "folHlink",
)


def _default_template_path() -> str:
    import pptx

    return os.path.join(os.path.dirname(pptx.__file__), "templates", "default.pptx")


def _patch_theme_xml(xml_bytes: bytes, theme: dict) -> bytes:
    root = etree.fromstring(xml_bytes)
    colors = theme["colors"]

    clr_scheme = root.find(".//a:clrScheme", _NS)
    for child in clr_scheme:
        slot = etree.QName(child).localname
        if slot not in _SLOT_KEYS:
            continue
        hex_val = colors[slot].lstrip("#").upper()
        for sub in list(child):
            child.remove(sub)
        srgb = etree.SubElement(child, f"{{{_A}}}srgbClr")
        srgb.set("val", hex_val)

    font_scheme = root.find(".//a:fontScheme", _NS)
    major = font_scheme.find("a:majorFont/a:latin", _NS)
    minor = font_scheme.find("a:minorFont/a:latin", _NS)
    if major is not None:
        major.set("typeface", theme["fonts"]["headline"])
    if minor is not None:
        minor.set("typeface", theme["fonts"]["body"])

    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def build_template(theme: dict | None = None) -> bytes:
    """Return branded template ``.pptx`` bytes."""
    theme = theme or WILFRED_BASE_THEME
    src = _default_template_path()
    with zipfile.ZipFile(src) as zin:
        entries = zin.infolist()
        payloads = {e.filename: zin.read(e.filename) for e in entries}

    payloads["ppt/theme/theme1.xml"] = _patch_theme_xml(
        payloads["ppt/theme/theme1.xml"], theme
    )

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for entry in entries:
            zout.writestr(entry, payloads[entry.filename])
    return out.getvalue()
