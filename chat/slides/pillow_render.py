"""Render a deck JSON straight to PNGs with Pillow — no LibreOffice, no browser.

We generate every deck from our own constrained schema (text / shape / image /
table / line, absolute-positioned in points), so we don't need a general-purpose
PowerPoint renderer. This module draws that schema onto a Pillow canvas using the
shipped metric-compatible fonts (Carlito ≈ Calibri, Caladea ≈ Cambria), so a deck
renders **identically on any platform** (Windows dev and Heroku) with only
Pillow — already a dependency, tiny footprint.

The same PNGs feed both the agent's `slides_preview_slide` visual loop and the
user's filmstrip. The downloadable ``.pptx`` stays python-pptx (``pptx_build``);
this is a *preview* renderer, kept close to PowerPoint via the metric fonts.

Pure Python (Pillow + the theme module) — unit-testable and runnable anywhere.
Coordinates are points; ``dpi`` sets the raster scale (default 120 → 1600×900 for
a 960×540 deck), matching the old LibreOffice→pypdfium2 pipeline.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from chat.slides import theme as theme_mod

logger = logging.getLogger(__name__)

# chat/slides/ → parents[2] is the repo root; core/ is a sibling of chat/.
_FONTS_DIR = Path(__file__).resolve().parents[2] / "core" / "assets" / "fonts"

_FALLBACK_FAMILY = "Carlito"
_DEFAULT_LINE_SPACING = 1.16   # PowerPoint single-spacing ≈ 1.15–1.2×
_PARA_GAP_DEFAULT = 0.0        # extra gap between paragraphs (pt) unless space_after set


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------
def _rgb(theme: dict, value, default=(0, 0, 0)) -> tuple[int, int, int]:
    """Resolve any colour reference (native slot / semantic name / #hex) to RGB."""
    if not value:
        return default
    hexv = value if isinstance(value, str) and value.startswith("#") else theme["colors"].get(value)
    if not (isinstance(hexv, str) and hexv.startswith("#")):
        return default
    h = hexv.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
@lru_cache(maxsize=512)
def _font(family: str, px: int, bold: bool, italic: bool) -> ImageFont.FreeTypeFont:
    px = max(1, int(round(px)))
    face = ("Bold" if bold else "") + ("Italic" if italic else "") or "Regular"
    candidates = [
        _FONTS_DIR / family / f"{family}-{face}.ttf",
        _FONTS_DIR / family / f"{family}-Regular.ttf",
        _FONTS_DIR / _FALLBACK_FAMILY / f"{_FALLBACK_FAMILY}-{face}.ttf",
        _FONTS_DIR / _FALLBACK_FAMILY / f"{_FALLBACK_FAMILY}-Regular.ttf",
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), px)
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default()


def _run_font(theme: dict, style: dict, scale: float) -> ImageFont.FreeTypeFont:
    family = theme_mod.font_family(theme, style.get("font"))
    size_pt = style.get("size") or 14
    return _font(family, size_pt * scale, bool(style.get("bold")), bool(style.get("italic")))


# ---------------------------------------------------------------------------
# Text layout
#   A paragraph's runs are tokenised into styled words, greedily packed into
#   lines that fit the box width, then drawn with the paragraph's alignment and
#   the text-frame's vertical anchor. Line height tracks the tallest run.
# ---------------------------------------------------------------------------
class _Word:
    __slots__ = ("text", "font", "color", "underline", "w", "ascent", "descent", "space_w")

    def __init__(self, text, font, color, underline):
        self.text = text
        self.font = font
        self.color = color
        self.underline = underline
        self.w = font.getlength(text)
        self.ascent, self.descent = font.getmetrics()
        self.space_w = font.getlength(" ")


def _para_words(theme, para, base_style, scale):
    """Flatten a paragraph's runs into styled words (space-separated tokens).

    Returns ``(words, forced_breaks)`` where ``forced_breaks`` marks indices
    after which a hard line break occurs (explicit ``\\n`` in run text).
    """
    words: list[_Word] = []
    breaks: set[int] = set()
    for run in para.get("runs") or []:
        style = theme_mod.merge_run(base_style, run)
        font = _run_font(theme, style, scale)
        color = _rgb(theme, style.get("color"), (0, 0, 0))
        underline = bool(style.get("underline"))
        text = run.get("t", "")
        segments = text.split("\n")
        for si, seg in enumerate(segments):
            for tok in seg.split(" "):
                if tok == "":
                    continue
                words.append(_Word(tok, font, color, underline))
            if si < len(segments) - 1 and words:
                breaks.add(len(words) - 1)  # hard break after the last word so far
    return words, breaks


def _wrap(words, breaks, max_w):
    """Greedily pack words into lines that fit ``max_w`` px. Returns list of lines;
    each line is a list of ``(word, x_offset)`` plus its width and height."""
    lines = []
    cur, cur_w = [], 0.0
    for i, word in enumerate(words):
        add = word.w if not cur else word.space_w + word.w
        if cur and cur_w + add > max_w:
            lines.append(cur)
            cur, cur_w = [], 0.0
            add = word.w
        cur.append(word)
        cur_w += add
        if i in breaks:
            lines.append(cur)
            cur, cur_w = [], 0.0
    if cur:
        lines.append(cur)
    return lines


def _line_width(line):
    w = 0.0
    for i, word in enumerate(line):
        w += word.w if i == 0 else word.space_w + word.w
    return w


def _draw_text_frame(draw, theme, frame, box, scale):
    """Render a text frame (paragraphs) inside ``box`` (x, y, w, h in px)."""
    bx, by, bw, bh = box
    default_cls = frame.get("class")
    paragraphs = frame.get("paragraphs") or []
    valign = frame.get("valign", "top")
    bullet_char = (theme.get("bullet", {}) or {}).get("char", "•")

    # First pass: lay out every paragraph into positioned lines to get total height.
    laid = []  # list of dicts: {lines, align, indent, bullet, para_gap, line_h}
    total_h = 0.0
    for para in paragraphs:
        base_style = theme_mod.base_text_style(theme, para.get("class") or default_cls)
        words, breaks = _para_words(theme, para, base_style, scale)
        level = int(para.get("level", 0) or 0)
        is_bullet = bool(para.get("bullet"))
        # Bullet: PowerPoint hangs the text at a fixed indent (~0.3") from the
        # bullet glyph, which sits at the (level-nested) left edge. Match that so
        # wrapped lines align under the text, not the bullet.
        level_indent = level * 20 * scale
        hang = 22 * scale
        indent = level_indent
        bullet_glyph = None
        if is_bullet:
            bfont = _run_font(theme, base_style, scale)
            bullet_glyph = (bullet_char, bfont, _rgb(theme, base_style.get("color"), (0, 0, 0)), level_indent)
            indent = level_indent + hang
        avail = max(1.0, bw - indent)
        lines = _wrap(words, breaks, avail) if words else [[]]
        # line height from the tallest run on each line (fallback to base size)
        base_px = (base_style.get("size") or 14) * scale
        para_line_h = []
        for line in lines:
            if line:
                asc = max(w.ascent for w in line)
                desc = max(w.descent for w in line)
                lh = (asc + desc) * _DEFAULT_LINE_SPACING
            else:
                lh = base_px * _DEFAULT_LINE_SPACING
            para_line_h.append(lh)
        para_h = sum(para_line_h)
        space_after = para.get("space_after")
        gap = (space_after if space_after is not None else _PARA_GAP_DEFAULT) * scale
        laid.append({
            "lines": lines, "line_h": para_line_h, "align": para.get("align", "left"),
            "indent": indent, "bullet": bullet_glyph, "gap": gap,
        })
        total_h += para_h + gap

    # Vertical anchor.
    if valign == "middle":
        cursor_y = by + max(0.0, (bh - total_h) / 2)
    elif valign == "bottom":
        cursor_y = by + max(0.0, bh - total_h)
    else:
        cursor_y = by

    # Second pass: draw.
    for p in laid:
        for li, line in enumerate(p["lines"]):
            lh = p["line_h"][li]
            line_w = _line_width(line)
            x0 = bx + p["indent"]
            if p["align"] == "center":
                x0 = bx + p["indent"] + max(0.0, (bw - p["indent"] - line_w) / 2)
            elif p["align"] == "right":
                x0 = bx + bw - line_w
            baseline = cursor_y + (lh / _DEFAULT_LINE_SPACING) - (max((w.descent for w in line), default=0))
            # Bullet glyph on the first line of the paragraph, at its level edge.
            if li == 0 and p["bullet"]:
                bchar, bfont, bcolor, blevel_x = p["bullet"]
                draw.text((bx + blevel_x, cursor_y), bchar, font=bfont, fill=bcolor)
            x = x0
            for wi, word in enumerate(line):
                if wi > 0:
                    x += word.space_w
                draw.text((x, cursor_y), word.text, font=word.font, fill=word.color)
                if word.underline:
                    uy = cursor_y + word.ascent + max(1, int(scale))
                    draw.line([(x, uy), (x + word.w, uy)], fill=word.color, width=max(1, int(scale)))
                x += word.w
            cursor_y += lh
        cursor_y += p["gap"]


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------
def _draw_shape(draw, theme, el, scale):
    x, y, w, h = (el["x"] * scale, el["y"] * scale, el["w"] * scale, el["h"] * scale)
    box = theme.get("boxes", {}).get(el["box"]) if el.get("box") else None
    box = box or {}
    name = el.get("shape") or box.get("shape") or "rect"
    fill_v = el.get("fill") if el.get("fill") is not None else box.get("fill")
    fill = _rgb(theme, fill_v, None) if fill_v is not None else None
    line = el.get("line") or {}
    outline = _rgb(theme, line.get("color"), None) if line.get("color") else None
    width = max(1, int((line.get("w") or 1) * scale)) if outline else 0
    x2, y2 = x + w, y + h

    if name in ("oval", "ellipse", "circle"):
        draw.ellipse([x, y, x2, y2], fill=fill, outline=outline, width=width)
    elif name in ("rounded_rect",):
        r = min(w, h) * 0.14
        draw.rounded_rectangle([x, y, x2, y2], radius=r, fill=fill, outline=outline, width=width)
    elif name in ("right_arrow", "left_arrow"):
        _draw_block_arrow(draw, x, y, w, h, name, fill, outline, width)
    elif name in ("chevron",):
        _draw_chevron(draw, x, y, w, h, fill, outline, width)
    elif name in ("diamond",):
        draw.polygon([(x + w / 2, y), (x2, y + h / 2), (x + w / 2, y2), (x, y + h / 2)],
                     fill=fill, outline=outline, width=width)
    else:  # rect + any unmapped shape
        draw.rectangle([x, y, x2, y2], fill=fill, outline=outline, width=width)

    text = el.get("text")
    if text:
        default_color = box.get("text_color")
        frame = {
            "class": text.get("class") or box.get("class"),
            "paragraphs": _inject_default_color(text.get("paragraphs"), default_color),
            "valign": text.get("valign", "middle"),
        }
        pad = 6 * scale
        _draw_text_frame(draw, theme, frame, (x + pad, y + pad, w - 2 * pad, h - 2 * pad), scale)


def _inject_default_color(paragraphs, default_color):
    if not default_color or not paragraphs:
        return paragraphs
    out = []
    for para in paragraphs:
        p = dict(para)
        p["runs"] = [(r if r.get("color") else {**r, "color": default_color}) for r in (para.get("runs") or [])]
        out.append(p)
    return out


def _draw_block_arrow(draw, x, y, w, h, name, fill, outline, width):
    head = min(w * 0.4, h)  # arrowhead length
    shaft = h * 0.5
    sy0, sy1 = y + (h - shaft) / 2, y + (h + shaft) / 2
    if name == "right_arrow":
        pts = [(x, sy0), (x + w - head, sy0), (x + w - head, y), (x + w, y + h / 2),
               (x + w - head, y + h), (x + w - head, sy1), (x, sy1)]
    else:  # left_arrow
        pts = [(x + w, sy0), (x + head, sy0), (x + head, y), (x, y + h / 2),
               (x + head, y + h), (x + head, sy1), (x + w, sy1)]
    draw.polygon(pts, fill=fill, outline=outline, width=width)


def _draw_chevron(draw, x, y, w, h, fill, outline, width):
    notch = w * 0.28
    pts = [(x, y), (x + w - notch, y), (x + w, y + h / 2), (x + w - notch, y + h),
           (x, y + h), (x + notch, y + h / 2)]
    draw.polygon(pts, fill=fill, outline=outline, width=width)


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------
def _draw_line(draw, theme, el, scale):
    x1, y1 = el["x1"] * scale, el["y1"] * scale
    x2, y2 = el["x2"] * scale, el["y2"] * scale
    color = _rgb(theme, el.get("color"), (60, 60, 60))
    width = max(1, int((el.get("w") or 1) * scale))
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    arrow = el.get("arrow", "none")
    if arrow in ("end", "both"):
        _arrowhead(draw, x1, y1, x2, y2, color, width)
    if arrow in ("start", "both"):
        _arrowhead(draw, x2, y2, x1, y1, color, width)


def _arrowhead(draw, fx, fy, tx, ty, color, width):
    ang = math.atan2(ty - fy, tx - fx)
    size = 6 + width * 2
    for da in (math.radians(150), math.radians(-150)):
        ex = tx + size * math.cos(ang + da)
        ey = ty + size * math.sin(ang + da)
        draw.line([(tx, ty), (ex, ey)], fill=color, width=width)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _draw_table(draw, theme, el, scale):
    rows = el.get("rows") or []
    if not rows:
        return
    n_cols = max(len(r) for r in rows)
    x, y, w = el["x"] * scale, el["y"] * scale, el["w"] * scale
    col_widths = el.get("col_widths")
    if col_widths and len(col_widths) == n_cols:
        cw = [c * scale for c in col_widths]
        # normalise to the element width
        s = sum(cw)
        cw = [c * w / s for c in cw] if s else [w / n_cols] * n_cols
    else:
        cw = [w / n_cols] * n_cols

    tbl = theme.get("table", {})
    text_class = tbl.get("text_class", "data")
    header = bool(el.get("header", True))
    banding = bool(el.get("banding", True))
    data_size = (theme.get("text_styles", {}).get(text_class, {}).get("size", 12))
    row_h = max(26, data_size * 2.0) * scale

    cy = y
    for ri, row in enumerate(rows):
        is_header = header and ri == 0
        is_band = banding and not is_header and (ri % 2 == 0)
        cx = x
        for ci in range(n_cols):
            spec = row[ci] if ci < len(row) else {}
            spec = spec or {}
            cell_w = cw[ci]
            # fill
            fill_v = spec.get("fill")
            if fill_v is None:
                if is_header:
                    fill_v = tbl.get("header_fill")
                elif is_band:
                    fill_v = tbl.get("band_fill")
            if fill_v:
                draw.rectangle([cx, cy, cx + cell_w, cy + row_h], fill=_rgb(theme, fill_v, None))
            # text
            base = theme_mod.base_text_style(theme, spec.get("class") or text_class)
            color = spec.get("color") or (tbl.get("header_color") if is_header else base.get("color"))
            run = {"t": spec.get("t", "")}
            if spec.get("b") is not None:
                run["b"] = spec["b"]
            elif is_header:
                run["b"] = True
            if spec.get("i") is not None:
                run["i"] = spec["i"]
            if spec.get("size"):
                run["size"] = spec["size"]
            run["color"] = color
            frame = {"paragraphs": [{"runs": [run], "align": spec.get("align", "left")}],
                     "valign": "middle"}
            pad = 8 * scale
            _draw_text_frame(draw, theme, {**frame, "class": spec.get("class") or text_class},
                             (cx + pad, cy, cell_w - 2 * pad, row_h), scale)
            cx += cell_w
        cy += row_h
    # No grid lines: the .pptx builder neutralises the table style and relies on
    # header fill + banding only, so a borderless render matches the download.


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def _draw_image(img, theme, el, scale, resolver):
    x, y, w, h = int(el["x"] * scale), int(el["y"] * scale), int(el["w"] * scale), int(el["h"] * scale)
    token = el.get("token", "")
    data = resolver(token) if (resolver and token) else None
    if not data:
        _draw_placeholder(img, x, y, w, h, empty=(not token), scale=scale, theme=theme)
        return
    try:
        src = Image.open(BytesIO(data[0])).convert("RGBA")
    except Exception:  # noqa: BLE001
        _draw_placeholder(img, x, y, w, h, empty=False, scale=scale, theme=theme)
        return
    fit = el.get("fit", "contain")
    if fit == "stretch":
        img.paste(src.resize((w, h)), (x, y), src.resize((w, h)))
        return
    sw, sh = src.size
    if not sw or not sh:
        return
    if fit == "cover":
        s = max(w / sw, h / sh)
        nw, nh = int(sw * s), int(sh * s)
        resized = src.resize((nw, nh))
        left, top = (nw - w) // 2, (nh - h) // 2
        cropped = resized.crop((left, top, left + w, top + h))
        img.paste(cropped, (x, y), cropped)
    else:  # contain
        s = min(w / sw, h / sh)
        nw, nh = max(1, int(sw * s)), max(1, int(sh * s))
        resized = src.resize((nw, nh))
        img.paste(resized, (x + (w - nw) // 2, y + (h - nh) // 2), resized)


def _draw_placeholder(img, x, y, w, h, *, empty, scale, theme):
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([x, y, x + w, y + h], radius=8 * scale,
                        fill=(236, 239, 233), outline=(199, 207, 196), width=max(1, int(scale)))
    label = "Add an image" if empty else "image unavailable"
    font = _font("Carlito", 13 * scale, False, False)
    tw = font.getlength(label)
    d.text((x + (w - tw) / 2, y + h / 2 - 8 * scale), label, font=font, fill=(138, 148, 139))


# ---------------------------------------------------------------------------
# Footer / page number
# ---------------------------------------------------------------------------
def _stamp_footer(draw, theme, scale, page_num, total):
    footer = theme.get("footer", {})
    if footer.get("text"):
        font = _font(theme_mod.font_family(theme, "data"), (footer.get("size", 9)) * scale, False, False)
        draw.text((footer["x"] * scale, footer["y"] * scale), footer["text"],
                  font=font, fill=_rgb(theme, footer.get("color", "dk2")))
    pn = theme.get("page_number", {})
    font = _font(theme_mod.font_family(theme, pn.get("font", "data")), (pn.get("size", 9)) * scale, False, False)
    txt = str(page_num)
    tw = font.getlength(txt)
    px = pn.get("x", 900) * scale + (pn.get("w", 36) * scale - tw)  # right-align in the box
    draw.text((px, pn.get("y", 512) * scale), txt, font=font, fill=_rgb(theme, pn.get("color", "dk2")))


# ---------------------------------------------------------------------------
# Slide / deck
# ---------------------------------------------------------------------------
def render_slide_png(deck: dict, index: int, *, dpi: int = 120, image_resolver=None,
                     supersample: int = 2) -> tuple[bytes, int, int]:
    """Render slide ``index`` of ``deck`` to ``(png_bytes, width_px, height_px)``."""
    theme = theme_mod.resolve_theme(deck)
    size = deck.get("size") or {}
    w_pt, h_pt = size.get("w", 960), size.get("h", 540)
    scale = (dpi / 72.0) * supersample
    W, H = int(round(w_pt * scale)), int(round(h_pt * scale))
    slides = deck.get("slides") or []
    slide = slides[index]

    bg = _rgb(theme, slide.get("bg") or "lt1", (255, 255, 255))
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    for el in slide.get("elements") or []:
        etype = el.get("type")
        try:
            if etype == "text":
                frame = {"class": el.get("class"), "paragraphs": el.get("paragraphs"),
                         "valign": el.get("valign", "top")}
                box = (el["x"] * scale, el["y"] * scale, el["w"] * scale, el["h"] * scale)
                _draw_text_frame(draw, theme, frame, box, scale)
            elif etype == "shape":
                _draw_shape(draw, theme, el, scale)
            elif etype == "image":
                _draw_image(img, theme, el, scale, image_resolver)
            elif etype == "table":
                _draw_table(draw, theme, el, scale)
            elif etype == "line":
                _draw_line(draw, theme, el, scale)
        except Exception:  # noqa: BLE001 — one bad element must not fail the slide
            logger.warning("pillow render: element %s failed", etype, exc_info=True)

    if not slide.get("skip_footer"):
        _stamp_footer(draw, theme, scale, index + 1, len(slides))

    if supersample > 1:
        img = img.resize((int(round(w_pt * dpi / 72.0)), int(round(h_pt * dpi / 72.0))), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), img.width, img.height


def render_deck_pngs(deck: dict, *, dpi: int = 120, image_resolver=None,
                     only_slide_ids=None) -> list[tuple[str, bytes, int, int]]:
    """Render a deck to ``[(slide_id, png_bytes, w, h)]``. ``only_slide_ids`` limits it."""
    slides = deck.get("slides") or []
    only = set(only_slide_ids) if only_slide_ids else None
    out = []
    for i, s in enumerate(slides):
        sid = s.get("id") or f"_{i}"
        if only is not None and sid not in only:
            continue
        png, w, h = render_slide_png(deck, i, dpi=dpi, image_resolver=image_resolver)
        out.append((sid, png, w, h))
    return out
