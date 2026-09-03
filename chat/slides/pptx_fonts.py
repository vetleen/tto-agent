"""Embed font faces into a built ``.pptx`` (OOXML font embedding).

python-pptx has no API for embedded fonts, so we post-process the saved package:
add ``ppt/fonts/fontN.fntdata`` parts (raw TTF/OTF bytes), a content-type default,
presentation-part relationships, and a ``<p:embeddedFontLst>`` (plus
``embedTrueTypeFonts``) on ``ppt/presentation.xml``. Only used for **uploaded/custom**
families — bundled families reference metric-compatible Office fonts by name and are
not embedded (see ``_PPTX_FONT_ALIAS`` in :mod:`chat.slides.pptx_build`).

``embed_fonts(pptx_bytes, families) -> bytes`` where ``families`` is
``[{"typeface": str, "faces": [core.fonts.FontFace, …]}, …]``. The ``typeface`` MUST
equal the run font name stamped in the deck so PowerPoint matches the embedded face.
Never raises on a single bad face; returns the input bytes unchanged on hard failure.
"""
from __future__ import annotations

import logging
import re
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED

logger = logging.getLogger(__name__)

_P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
_R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_FONT_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/font"
_FONT_CONTENT_TYPE = "application/x-fontdata"

_PRESENTATION = "ppt/presentation.xml"
_PRES_RELS = "ppt/_rels/presentation.xml.rels"
_CONTENT_TYPES = "[Content_Types].xml"

# CT_Presentation child order — embeddedFontLst must sit right after notesSz (the
# preceding elements smartTags/notesSz/sldSz are what we anchor against).
_AFTER = ("notesSz", "sldSz", "notesMasterIdLst", "sldMasterIdLst")


def _face_slot(face) -> str:
    bold = int(getattr(face, "weight", 400) or 400) >= 700
    italic = getattr(face, "style", "normal") == "italic"
    if bold and italic:
        return "boldItalic"
    if bold:
        return "bold"
    if italic:
        return "italic"
    return "regular"


def embed_fonts(pptx_bytes: bytes, families: list[dict]) -> bytes:
    if not families:
        return pptx_bytes
    try:
        from lxml import etree
    except Exception:  # noqa: BLE001
        logger.warning("lxml unavailable; skipping .pptx font embedding")
        return pptx_bytes

    try:
        with ZipFile(BytesIO(pptx_bytes)) as zin:
            names = zin.namelist()
            entries = {n: zin.read(n) for n in names}
    except Exception:  # noqa: BLE001
        logger.warning("could not open .pptx for font embedding", exc_info=True)
        return pptx_bytes

    if _PRESENTATION not in entries or _PRES_RELS not in entries or _CONTENT_TYPES not in entries:
        return pptx_bytes

    try:
        pres = etree.fromstring(entries[_PRESENTATION])
        rels = etree.fromstring(entries[_PRES_RELS])
        cts = etree.fromstring(entries[_CONTENT_TYPES])
    except Exception:  # noqa: BLE001
        logger.warning("could not parse .pptx parts for font embedding", exc_info=True)
        return pptx_bytes

    # Next free relationship id + font-part index (avoid clobbering existing).
    used_ids = {r.get("Id") for r in rels}
    next_rid = 1 + max((int(m.group(1)) for r in rels
                        for m in [re.fullmatch(r"rId(\d+)", r.get("Id") or "")] if m), default=0)
    font_idx = 1 + max((int(m.group(1)) for n in names
                       for m in [re.fullmatch(r"ppt/fonts/font(\d+)\.fntdata", n)] if m), default=0)

    embedded = []  # (typeface, {slot: rId})
    for fam in families:
        typeface = (fam.get("typeface") or "").strip()
        faces = fam.get("faces") or []
        if not typeface or not faces:
            continue
        slots: dict[str, str] = {}
        for face in faces:
            data = getattr(face, "data", None)
            if not data:
                continue
            slot = _face_slot(face)
            if slot in slots:  # one face per slot (first wins)
                continue
            part = f"ppt/fonts/font{font_idx}.fntdata"
            font_idx += 1
            entries[part] = data
            rid = f"rId{next_rid}"
            while rid in used_ids:
                next_rid += 1
                rid = f"rId{next_rid}"
            used_ids.add(rid)
            next_rid += 1
            rel = etree.SubElement(rels, f"{{{_REL_NS}}}Relationship")
            rel.set("Id", rid)
            rel.set("Type", _FONT_REL_TYPE)
            rel.set("Target", f"fonts/font{font_idx - 1}.fntdata")
            slots[slot] = rid
        if slots:
            embedded.append((typeface, slots))

    if not embedded:
        return pptx_bytes

    # Content-types: ensure a default for the fntdata extension.
    if not any(el.get("Extension") == "fntdata" for el in cts
               if el.tag == f"{{{_CT_NS}}}Default"):
        default = etree.SubElement(cts, f"{{{_CT_NS}}}Default")
        default.set("Extension", "fntdata")
        default.set("ContentType", _FONT_CONTENT_TYPE)

    # presentation.xml: embedTrueTypeFonts + embeddedFontLst (after notesSz).
    pres.set("embedTrueTypeFonts", "1")
    lst = etree.Element(f"{{{_P_NS}}}embeddedFontLst")
    for typeface, slots in embedded:
        efont = etree.SubElement(lst, f"{{{_P_NS}}}embeddedFont")
        font_el = etree.SubElement(efont, f"{{{_P_NS}}}font")
        font_el.set("typeface", typeface)
        for slot in ("regular", "bold", "italic", "boldItalic"):
            if slot in slots:
                child = etree.SubElement(efont, f"{{{_P_NS}}}{slot}")
                child.set(f"{{{_R_NS}}}id", slots[slot])
    _insert_after(pres, lst)

    entries[_PRESENTATION] = _ser(etree, pres)
    entries[_PRES_RELS] = _ser(etree, rels)
    entries[_CONTENT_TYPES] = _ser(etree, cts)

    out = BytesIO()
    with ZipFile(out, "w", ZIP_DEFLATED) as zout:
        for name, data in entries.items():
            zout.writestr(name, data)
    return out.getvalue()


def _insert_after(pres, lst) -> None:
    """Insert ``lst`` into ``<p:presentation>`` right after the first of ``_AFTER``
    found (schema order), else append."""
    children = list(pres)
    for local in _AFTER:
        for i, child in enumerate(children):
            if child.tag == f"{{{_P_NS}}}{local}":
                pres.insert(i + 1, lst)
                return
    pres.append(lst)


def _ser(etree, el) -> bytes:
    return etree.tostring(el, xml_declaration=True, encoding="UTF-8", standalone=True)
