"""Deck JSON -> ``.pptx`` bytes (python-pptx).

``build_deck_pptx(deck, ...)`` is the pure core (no Django) so it unit-tests
with golden-OOXML assertions on any platform. ``build_pptx(slide_set)`` is the
Django-aware wrapper that pulls the deck + base template off a ``SlideSet`` and
binds an image resolver scoped to the deck's owner.

Coordinates are points (Pt); python-pptx converts to EMU. Colours resolve via
:mod:`chat.slides.theme` — native theme slots become ``MSO_THEME_COLOR`` refs
(tracking the template's scheme + showing in PowerPoint's picker), everything
else an explicit RGB.
"""

from __future__ import annotations

import logging
import re
from io import BytesIO
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE, MSO_THEME_COLOR
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Pt

from chat.slides import oxml_pokes as pokes
from chat.slides import theme as theme_mod

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
TEMPLATE_DIR = ASSETS_DIR / "templates"

# Curated shape vocabulary (friendly name -> MSO_SHAPE). Verified members only;
# unknown names fall back to RECTANGLE with a warning.
SHAPE_MAP = {
    "rect": MSO_SHAPE.RECTANGLE,
    "rectangle": MSO_SHAPE.RECTANGLE,
    "rounded_rect": MSO_SHAPE.ROUNDED_RECTANGLE,
    "oval": MSO_SHAPE.OVAL,
    "ellipse": MSO_SHAPE.OVAL,
    "circle": MSO_SHAPE.OVAL,
    "right_arrow": MSO_SHAPE.RIGHT_ARROW,
    "left_arrow": MSO_SHAPE.LEFT_ARROW,
    "up_arrow": MSO_SHAPE.UP_ARROW,
    "down_arrow": MSO_SHAPE.DOWN_ARROW,
    "chevron": MSO_SHAPE.CHEVRON,
    "pentagon": MSO_SHAPE.PENTAGON,
    "diamond": MSO_SHAPE.DIAMOND,
    "hexagon": MSO_SHAPE.HEXAGON,
    "star": MSO_SHAPE.STAR_5_POINT,
    "plus": MSO_SHAPE.MATH_PLUS,
    "cross": MSO_SHAPE.CROSS,
    "cloud": MSO_SHAPE.CLOUD,
    "heart": MSO_SHAPE.HEART,
    "donut": MSO_SHAPE.DONUT,
    "plaque": MSO_SHAPE.PLAQUE,
    "frame": MSO_SHAPE.FRAME,
    "bevel": MSO_SHAPE.BEVEL,
    "cube": MSO_SHAPE.CUBE,
    "can": MSO_SHAPE.CAN,
    "callout": MSO_SHAPE.LINE_CALLOUT_1,
    "block_arc": MSO_SHAPE.BLOCK_ARC,
    "tear": MSO_SHAPE.TEAR,
    "lightning": MSO_SHAPE.LIGHTNING_BOLT,
    "smiley": MSO_SHAPE.SMILEY_FACE,
}

THEME_COLOR_MAP = {
    "dk1": MSO_THEME_COLOR.DARK_1,
    "lt1": MSO_THEME_COLOR.LIGHT_1,
    "dk2": MSO_THEME_COLOR.DARK_2,
    "lt2": MSO_THEME_COLOR.LIGHT_2,
    "accent1": MSO_THEME_COLOR.ACCENT_1,
    "accent2": MSO_THEME_COLOR.ACCENT_2,
    "accent3": MSO_THEME_COLOR.ACCENT_3,
    "accent4": MSO_THEME_COLOR.ACCENT_4,
    "accent5": MSO_THEME_COLOR.ACCENT_5,
    "accent6": MSO_THEME_COLOR.ACCENT_6,
    "hlink": MSO_THEME_COLOR.HYPERLINK,
    "folHlink": MSO_THEME_COLOR.FOLLOWED_HYPERLINK,
}

ALIGN_MAP = {
    "left": PP_ALIGN.LEFT,
    "center": PP_ALIGN.CENTER,
    "right": PP_ALIGN.RIGHT,
    "justify": PP_ALIGN.JUSTIFY,
}

ANCHOR_MAP = {
    "top": MSO_ANCHOR.TOP,
    "middle": MSO_ANCHOR.MIDDLE,
    "bottom": MSO_ANCHOR.BOTTOM,
}

DASH_MAP = {
    "solid": MSO_LINE_DASH_STYLE.SOLID,
    "dash": MSO_LINE_DASH_STYLE.DASH,
    "dot": MSO_LINE_DASH_STYLE.ROUND_DOT,
    "dashdot": MSO_LINE_DASH_STYLE.DASH_DOT,
}

_TOKEN_RE = re.compile(r"\[\[image:([0-9a-fA-F-]{36})")


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
def _apply_color(color_format, theme: dict, value) -> None:
    """Apply a resolved colour onto a python-pptx ColorFormat (font/fill/line)."""
    resolved = theme_mod.resolve_color(theme, value)
    if resolved is None:
        return
    kind, val = resolved
    if kind == "theme":
        color_format.theme_color = THEME_COLOR_MAP[val]
    else:
        color_format.rgb = RGBColor.from_string(val)


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
def _render_text_body(text_frame, body: dict, theme: dict, *, default_class: str | None = None) -> None:
    text_frame.word_wrap = bool(body.get("wrap", True))
    text_frame.vertical_anchor = ANCHOR_MAP.get(body.get("valign", "top"), MSO_ANCHOR.TOP)
    cls = body.get("class") or default_class
    base = theme_mod.base_text_style(theme, cls)

    paragraphs = body.get("paragraphs") or []
    text_frame.clear()  # leaves a single empty paragraph
    for pi, para in enumerate(paragraphs):
        p = text_frame.paragraphs[0] if pi == 0 else text_frame.add_paragraph()
        p.alignment = ALIGN_MAP.get(para.get("align", "left"), PP_ALIGN.LEFT)
        if para.get("space_after") is not None:
            p.space_after = Pt(para["space_after"])
        level = int(para.get("level", 0) or 0)
        if level:
            p.level = level
        for run in para.get("runs") or []:
            r = p.add_run()
            r.text = run.get("t", "")
            style = theme_mod.merge_run(base, run)
            r.font.name = theme_mod.font_family(theme, style.get("font"))
            if style.get("size"):
                r.font.size = Pt(style["size"])
            r.font.bold = bool(style.get("bold"))
            r.font.italic = bool(style.get("italic"))
            if style.get("underline"):
                r.font.underline = True
            _apply_color(r.font.color, theme, style.get("color"))
        # Bullet handling is a poke (last, so it stays schema-valid).
        if para.get("bullet"):
            pokes.set_bullet_char(p, theme.get("bullet", {}).get("char", "•"))
        else:
            pokes.set_no_bullet(p)


# ---------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------
def _pt_box(el: dict):
    return Pt(el["x"]), Pt(el["y"]), Pt(el["w"]), Pt(el["h"])


def _add_text(slide, el, theme):
    x, y, w, h = _pt_box(el)
    tb = slide.shapes.add_textbox(x, y, w, h)
    body = {
        "class": el.get("class"),
        "paragraphs": el.get("paragraphs"),
        "valign": el.get("valign", "top"),
        "wrap": el.get("wrap", True),
    }
    _render_text_body(tb.text_frame, body, theme)
    if el.get("rotation"):
        tb.rotation = el["rotation"]


def _add_shape(slide, el, theme, warnings):
    box = theme.get("boxes", {}).get(el["box"]) if el.get("box") else None
    box = box or {}
    shape_name = el.get("shape") or box.get("shape") or "rect"
    mso = SHAPE_MAP.get(shape_name)
    if mso is None:
        warnings.append(f"unknown shape '{shape_name}' -> rectangle")
        mso = MSO_SHAPE.RECTANGLE
    x, y, w, h = _pt_box(el)
    shp = slide.shapes.add_shape(mso, x, y, w, h)

    fill_v = el.get("fill") if el.get("fill") is not None else box.get("fill")
    if fill_v is not None:
        shp.fill.solid()
        _apply_color(shp.fill.fore_color, theme, fill_v)
    else:
        shp.fill.background()

    line_spec = el.get("line")
    if line_spec:
        if line_spec.get("color"):
            _apply_color(shp.line.color, theme, line_spec["color"])
        if line_spec.get("w"):
            shp.line.width = Pt(line_spec["w"])
        if line_spec.get("dash"):
            shp.line.dash_style = DASH_MAP.get(line_spec["dash"], MSO_LINE_DASH_STYLE.SOLID)
    else:
        shp.line.fill.background()

    text = el.get("text")
    if text:
        default_class = box.get("class")
        # A box's text_color is the default run colour unless the text overrides.
        if box.get("text_color") and not _text_has_color(text):
            text = dict(text)
            text.setdefault("_default_color", box["text_color"])
        _render_shape_text(shp.text_frame, text, theme, default_class, box.get("text_color"))
    if el.get("rotation"):
        shp.rotation = el["rotation"]


def _text_has_color(text: dict) -> bool:
    for para in text.get("paragraphs") or []:
        for run in para.get("runs") or []:
            if run.get("color"):
                return True
    return False


def _render_shape_text(text_frame, text, theme, default_class, default_color):
    # Reuse the text-body renderer but inject the box's default colour when the
    # runs don't set one, so a themed box's label is legible on its fill.
    body = {
        "class": text.get("class"),
        "paragraphs": _inject_default_color(text.get("paragraphs"), default_color),
        "valign": text.get("valign", "middle"),
        "wrap": text.get("wrap", True),
    }
    _render_text_body(text_frame, body, theme, default_class=default_class)


def _inject_default_color(paragraphs, default_color):
    if not default_color or not paragraphs:
        return paragraphs
    out = []
    for para in paragraphs:
        p = dict(para)
        p["runs"] = [
            (run if run.get("color") else {**run, "color": default_color})
            for run in para.get("runs") or []
        ]
        out.append(p)
    return out


def _add_image(slide, el, theme, resolver, warnings):
    x, y, w, h = _pt_box(el)
    data = resolver(el.get("token", "")) if resolver else None
    if not data:
        warnings.append(f"image unavailable: {el.get('token', '')[:60]}")
        ph = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
        ph.fill.solid()
        ph.fill.fore_color.rgb = RGBColor(0xE4, 0xE4, 0xDE)
        ph.line.color.rgb = RGBColor(0xB8, 0xB8, 0xAE)
        ph.text_frame.text = "image unavailable"
        ph.text_frame.paragraphs[0].runs[0].font.size = Pt(10)
        return
    img_bytes, _ct = data
    stream = BytesIO(img_bytes)
    fit = el.get("fit", "contain")
    if fit == "stretch":
        slide.shapes.add_picture(stream, x, y, width=w, height=h)
        return
    pic = slide.shapes.add_picture(stream, x, y)  # native size first
    nat_w, nat_h = pic.width, pic.height
    if not nat_w or not nat_h:
        return
    box_w, box_h = int(w), int(h)
    if fit == "cover":
        scale = max(box_w / nat_w, box_h / nat_h)
        disp_w, disp_h = nat_w * scale, nat_h * scale
        crop_x = (disp_w - box_w) / disp_w / 2 if disp_w > box_w else 0
        crop_y = (disp_h - box_h) / disp_h / 2 if disp_h > box_h else 0
        pic.crop_left = crop_x
        pic.crop_right = crop_x
        pic.crop_top = crop_y
        pic.crop_bottom = crop_y
        pic.left, pic.top, pic.width, pic.height = int(x), int(y), box_w, box_h
    else:  # contain
        scale = min(box_w / nat_w, box_h / nat_h)
        new_w, new_h = int(nat_w * scale), int(nat_h * scale)
        pic.width, pic.height = new_w, new_h
        pic.left = int(x) + (box_w - new_w) // 2
        pic.top = int(y) + (box_h - new_h) // 2


def _add_table(slide, el, theme, warnings):
    rows = el.get("rows") or []
    n_rows = len(rows)
    n_cols = max((len(r) for r in rows), default=0)
    if n_rows == 0 or n_cols == 0:
        warnings.append("empty table skipped")
        return
    x, y, w, h = _pt_box(el)
    gf = slide.shapes.add_table(n_rows, n_cols, x, y, w, h)
    table = gf.table
    # Neutralise the default style so our explicit fills render as specified.
    pokes.set_table_style(table, pokes.NO_STYLE_NO_GRID)
    table.first_row = False
    table.horz_banding = False

    col_widths = el.get("col_widths")
    if col_widths and len(col_widths) == n_cols:
        for i, cw in enumerate(col_widths):
            table.columns[i].width = Pt(cw)

    tbl_theme = theme.get("table", {})
    header = bool(el.get("header", True))
    banding = bool(el.get("banding", True))
    text_class = tbl_theme.get("text_class", "data")

    for ri, row in enumerate(rows):
        is_header = header and ri == 0
        is_band = banding and not is_header and (ri % 2 == 0)
        for ci in range(n_cols):
            spec = row[ci] if ci < len(row) else {}
            _render_cell(table.cell(ri, ci), spec or {}, theme, tbl_theme, text_class, is_header, is_band)


def _render_cell(cell, spec, theme, tbl_theme, text_class, is_header, is_band):
    tf = cell.text_frame
    tf.clear()
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = ALIGN_MAP.get(spec.get("align", "left"), PP_ALIGN.LEFT)
    r = p.add_run()
    r.text = spec.get("t", "")
    cls = spec.get("class") or text_class
    style = theme_mod.base_text_style(theme, cls)
    r.font.name = theme_mod.font_family(theme, style.get("font"))
    if style.get("size"):
        r.font.size = Pt(style["size"])
    bold = spec.get("b")
    r.font.bold = bool(is_header if bold is None else bold)
    if spec.get("i") is not None:
        r.font.italic = bool(spec["i"])
    color = spec.get("color") or (tbl_theme.get("header_color") if is_header else style.get("color"))
    _apply_color(r.font.color, theme, color)

    fill_v = spec.get("fill")
    if fill_v is None:
        if is_header:
            fill_v = tbl_theme.get("header_fill")
        elif is_band:
            fill_v = tbl_theme.get("band_fill")
    if fill_v:
        cell.fill.solid()
        _apply_color(cell.fill.fore_color, theme, fill_v)
    else:
        cell.fill.background()


def _add_line(slide, el, theme):
    conn = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Pt(el["x1"]), Pt(el["y1"]), Pt(el["x2"]), Pt(el["y2"])
    )
    lf = conn.line
    if el.get("color"):
        _apply_color(lf.color, theme, el["color"])
    lf.width = Pt(el.get("w") or 1)
    if el.get("dash"):
        lf.dash_style = DASH_MAP.get(el["dash"], MSO_LINE_DASH_STYLE.SOLID)
    pokes.set_connector_arrows(lf, el.get("arrow", "none"))


def _render_element(slide, el, theme, resolver, warnings):
    etype = el.get("type")
    try:
        if etype == "text":
            _add_text(slide, el, theme)
        elif etype == "shape":
            _add_shape(slide, el, theme, warnings)
        elif etype == "image":
            _add_image(slide, el, theme, resolver, warnings)
        elif etype == "table":
            _add_table(slide, el, theme, warnings)
        elif etype == "line":
            _add_line(slide, el, theme)
        else:
            warnings.append(f"unknown element type '{etype}' skipped")
    except Exception as exc:  # noqa: BLE001 — one bad element must not fail the deck
        logger.warning("slide element render failed (%s): %s", etype, exc, exc_info=True)
        warnings.append(f"element '{etype}' failed to render")


# ---------------------------------------------------------------------------
# Slide chrome
# ---------------------------------------------------------------------------
def _apply_slide_bg(slide, theme, value):
    fill = slide.background.fill
    fill.solid()
    _apply_color(fill.fore_color, theme, value)


def _stamp_footer(slide, theme, page_num, total):
    footer = theme.get("footer", {})
    # Optional footer logo.
    logo_name = footer.get("logo_asset")
    if logo_name:
        logo_path = ASSETS_DIR / logo_name
        if logo_path.exists():
            try:
                slide.shapes.add_picture(
                    str(logo_path), Pt(footer["x"]), Pt(footer["y"]),
                    height=Pt(footer.get("h", 18)),
                )
            except Exception:  # noqa: BLE001
                logger.warning("footer logo failed: %s", logo_name)
    # Optional footer text.
    if footer.get("text"):
        tb = slide.shapes.add_textbox(
            Pt(footer["x"] + (footer.get("h", 18) if logo_name else 0)),
            Pt(footer["y"]), Pt(footer.get("w", 200)), Pt(footer.get("h", 18)),
        )
        p = tb.text_frame.paragraphs[0]
        p.alignment = ALIGN_MAP.get(footer.get("align", "left"), PP_ALIGN.LEFT)
        r = p.add_run()
        r.text = footer["text"]
        r.font.size = Pt(footer.get("size", 9))
        r.font.name = theme_mod.font_family(theme, "data")
        _apply_color(r.font.color, theme, footer.get("color", "dk2"))
    # Page number.
    pn = theme.get("page_number", {})
    tb = slide.shapes.add_textbox(Pt(pn.get("x", 900)), Pt(pn.get("y", 512)), Pt(pn.get("w", 36)), Pt(pn.get("h", 18)))
    p = tb.text_frame.paragraphs[0]
    p.alignment = ALIGN_MAP.get(pn.get("align", "right"), PP_ALIGN.RIGHT)
    r = p.add_run()
    r.text = str(page_num)
    r.font.size = Pt(pn.get("size", 9))
    r.font.name = theme_mod.font_family(theme, pn.get("font", "data"))
    _apply_color(r.font.color, theme, pn.get("color", "dk2"))


# ---------------------------------------------------------------------------
# Presentation assembly
# ---------------------------------------------------------------------------
def _open_template(base_template: str) -> Presentation:
    path = TEMPLATE_DIR / f"{base_template}.pptx"
    if path.exists():
        try:
            return Presentation(str(path))
        except Exception:  # noqa: BLE001
            logger.warning("template %s unreadable; using python-pptx default", base_template)
    return Presentation()


def _blank_layout(prs: Presentation):
    # The stock template's layout[6] is "Blank"; guard for custom templates.
    layouts = prs.slide_layouts
    return layouts[6] if len(layouts) > 6 else layouts[-1]


def build_deck_pptx(
    deck: dict,
    *,
    base_template: str = "wilfred_default",
    only_slide_ids=None,
    image_resolver=None,
) -> tuple[bytes, list[str]]:
    """Render a deck dict to ``.pptx`` bytes. Returns ``(bytes, warnings)``.

    ``only_slide_ids`` renders just those slides (for the partial render cache);
    page numbers still reflect each slide's position in the FULL deck so partial
    renders match. ``image_resolver(token) -> (bytes, content_type) | None``
    supplies image bytes; a missing image degrades to a placeholder box.
    """
    warnings: list[str] = []
    theme = theme_mod.resolve_theme(deck)
    prs = _open_template(base_template)
    size = deck.get("size") or {}
    prs.slide_width = Pt(size.get("w", 960))
    prs.slide_height = Pt(size.get("h", 540))
    blank = _blank_layout(prs)

    slides = deck.get("slides") or []
    total = len(slides)
    only = set(only_slide_ids) if only_slide_ids else None

    for idx, sdict in enumerate(slides):
        if only is not None and sdict.get("id") not in only:
            continue
        slide = prs.slides.add_slide(blank)
        if sdict.get("bg"):
            _apply_slide_bg(slide, theme, sdict["bg"])
        for el in sdict.get("elements") or []:
            _render_element(slide, el, theme, image_resolver, warnings)
        if not sdict.get("skip_footer"):
            _stamp_footer(slide, theme, idx + 1, total)
        if sdict.get("notes"):
            slide.notes_slide.notes_text_frame.text = sdict["notes"]

    out = BytesIO()
    prs.save(out)
    return out.getvalue(), warnings


# ---------------------------------------------------------------------------
# Django-aware wrapper
# ---------------------------------------------------------------------------
def build_pptx(slide_set, *, only_slide_ids=None) -> tuple[bytes, list[str]]:
    """Build ``.pptx`` bytes for a ``SlideSet`` model instance."""
    deck = slide_set.content or {}
    resolver = make_image_resolver(slide_set)
    return build_deck_pptx(
        deck,
        base_template=slide_set.base_template or "wilfred_default",
        only_slide_ids=only_slide_ids,
        image_resolver=resolver,
    )


def make_image_resolver(slide_set):
    """A resolver bound to a deck's owner: only assets owned by this deck's
    thread or the deck itself resolve, so a model-supplied token can't leak
    another user's asset bytes. Caches per (uuid) within one build.
    """
    from django.db.models import Q

    from chat.assets import image_asset_source
    from chat.models import Asset

    cache: dict[str, tuple[bytes, str] | None] = {}

    def resolve(token: str):
        match = _TOKEN_RE.search(token or "")
        if not match:
            return None
        aid = match.group(1).lower()
        if aid in cache:
            return cache[aid]
        result = None
        try:
            asset = Asset.objects.filter(
                Q(pk=aid)
                & (Q(slide_set=slide_set) | Q(thread_id=slide_set.thread_id))
            ).first()
            if asset is not None:
                source, ct = image_asset_source(asset)
                if source is not None:
                    with source.open("rb") as fh:
                        result = (fh.read(), ct or "image/png")
        except Exception:  # noqa: BLE001
            logger.warning("image token resolve failed: %s", aid, exc_info=True)
        cache[aid] = result
        return result

    return resolve
