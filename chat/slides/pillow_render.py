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


# PowerPoint text-frame default insets (python-pptx add_textbox / add_shape use
# these when no explicit margins are set): lIns/rIns 0.1", tIns/bIns 0.05". We
# apply them so line wrapping and vertical position match the .pptx.
_INSET_LR = 7.2   # pt
_INSET_TB = 3.6   # pt


def _draw_text_frame(draw, theme, frame, box, scale):
    """Render a text frame (paragraphs) inside ``box`` (x, y, w, h in px).

    The box is the full element/shape/cell rect; the standard PowerPoint text
    insets are applied here so callers pass unpadded rects.
    """
    bx, by, bw, bh = box
    ins_lr, ins_tb = _INSET_LR * scale, _INSET_TB * scale
    bx, by = bx + ins_lr, by + ins_tb
    bw, bh = max(1.0, bw - 2 * ins_lr), max(1.0, bh - 2 * ins_tb)
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
            justify_extra = 0.0
            if p["align"] == "center":
                x0 = bx + p["indent"] + max(0.0, (bw - p["indent"] - line_w) / 2)
            elif p["align"] == "right":
                x0 = bx + bw - line_w
            elif p["align"] == "justify" and li < len(p["lines"]) - 1 and len(line) > 1:
                # Spread the slack across the gaps (the last line stays left).
                justify_extra = max(0.0, (bw - p["indent"] - line_w) / (len(line) - 1))
            # Bullet glyph on the first line of the paragraph, at its level edge.
            if li == 0 and p["bullet"]:
                bchar, bfont, bcolor, blevel_x = p["bullet"]
                draw.text((bx + blevel_x, cursor_y), bchar, font=bfont, fill=bcolor)
            x = x0
            for wi, word in enumerate(line):
                if wi > 0:
                    x += word.space_w + justify_extra
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
    dash = line.get("dash")
    dashed = bool(outline and dash and dash != "solid")
    # For a dashed border, fill the shape without a solid outline and stroke the
    # perimeter dashed afterwards.
    o = None if dashed else outline
    ow = 0 if dashed else width
    x2, y2 = x + w, y + h
    border_pts = None  # perimeter for a dashed stroke (None -> no dashed border)

    if name == "harvey":
        _draw_harvey(draw, x, y, x2, y2, el.get("value"),
                     fill or _rgb(theme, "dk2", (50, 50, 50)), scale)
        return

    if name in ("oval", "ellipse", "circle"):
        draw.ellipse([x, y, x2, y2], fill=fill, outline=o, width=ow)
    elif name == "rounded_rect":
        r = min(w, h) * 0.14
        draw.rounded_rectangle([x, y, x2, y2], radius=r, fill=fill, outline=o, width=ow)
        border_pts = [(x, y), (x2, y), (x2, y2), (x, y2)]  # approx (ignores rounding)
    elif name in _POLY_SHAPES:
        pts = _POLY_SHAPES[name](x, y, w, h)
        draw.polygon(pts, fill=fill, outline=o, width=ow)
        border_pts = pts
    else:  # rect + any unmapped shape
        draw.rectangle([x, y, x2, y2], fill=fill, outline=o, width=ow)
        border_pts = [(x, y), (x2, y), (x2, y2), (x, y2)]

    if dashed and border_pts:
        loop = border_pts + [border_pts[0]]
        for a, b in zip(loop, loop[1:]):
            _styled_line(draw, a, b, outline, width, dash)

    text = el.get("text")
    if text:
        default_color = box.get("text_color")
        frame = {
            "class": text.get("class") or box.get("class"),
            "paragraphs": _inject_default_color(text.get("paragraphs"), default_color),
            "valign": text.get("valign", "middle"),
        }
        _draw_text_frame(draw, theme, frame, (x, y, w, h), scale)


def _harvey_fraction(value):
    """Clamp to 0..1 and quantise to fifths (0/¼/½/¾/1) — the 5 Harvey-ball states."""
    frac = 0.0 if value is None else max(0.0, min(1.0, float(value)))
    return round(frac * 4) / 4


def _draw_harvey(draw, x0, y0, x1, y1, value, color, scale):
    """A Harvey ball: a full ring with the value-fraction filled clockwise from 12."""
    frac = _harvey_fraction(value)
    ring = max(1, int(round(1.4 * scale)))
    draw.ellipse([x0, y0, x1, y1], outline=color, width=ring)
    if frac >= 1.0:
        draw.ellipse([x0, y0, x1, y1], fill=color)
    elif frac > 0:
        inset = ring
        draw.pieslice([x0 + inset, y0 + inset, x1 - inset, y1 - inset],
                      start=-90, end=-90 + frac * 360, fill=color)


def _draw_icon(draw, theme, el, scale):
    from chat.slides import icons

    x, y, w, h = el["x"] * scale, el["y"] * scale, el["w"] * scale, el["h"] * scale
    color = _rgb(theme, el.get("color") or "dk2", (50, 50, 50))
    pad = min(w, h) * 0.08
    sw = max(1, int(round(min(w, h) * 0.09)))
    icons.draw_icon(draw, el.get("name") or "check", (x + pad, y + pad, x + w - pad, y + h - pad), color, sw)


def _inject_default_color(paragraphs, default_color):
    if not default_color or not paragraphs:
        return paragraphs
    out = []
    for para in paragraphs:
        p = dict(para)
        p["runs"] = [(r if r.get("color") else {**r, "color": default_color}) for r in (para.get("runs") or [])]
        out.append(p)
    return out


def _block_arrow_pts(x, y, w, h, name):
    head = min(w * 0.4, h)  # arrowhead length
    shaft = h * 0.5
    sy0, sy1 = y + (h - shaft) / 2, y + (h + shaft) / 2
    if name == "right_arrow":
        return [(x, sy0), (x + w - head, sy0), (x + w - head, y), (x + w, y + h / 2),
                (x + w - head, y + h), (x + w - head, sy1), (x, sy1)]
    return [(x + w, sy0), (x + head, sy0), (x + head, y), (x, y + h / 2),
            (x + head, y + h), (x + head, sy1), (x + w, sy1)]


def _chevron_pts(x, y, w, h):
    notch = w * 0.28
    return [(x, y), (x + w - notch, y), (x + w, y + h / 2), (x + w - notch, y + h),
            (x, y + h), (x + notch, y + h / 2)]


def _star_pts(x, y, w, h):
    cx, cy = x + w / 2, y + h / 2
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        f = 1.0 if i % 2 == 0 else 0.40
        pts.append((cx + (w / 2) * f * math.cos(ang), cy + (h / 2) * f * math.sin(ang)))
    return pts


def _hexagon_pts(x, y, w, h):
    return [(x, y + h / 2), (x + w * 0.25, y), (x + w * 0.75, y),
            (x + w, y + h / 2), (x + w * 0.75, y + h), (x + w * 0.25, y + h)]


def _pentagon_pts(x, y, w, h):  # MSO "home plate" pentagon, pointing right
    return [(x, y), (x + w * 0.55, y), (x + w, y + h / 2), (x + w * 0.55, y + h), (x, y + h)]


def _diamond_pts(x, y, w, h):
    return [(x + w / 2, y), (x + w, y + h / 2), (x + w / 2, y + h), (x, y + h / 2)]


_POLY_SHAPES = {
    "diamond": _diamond_pts, "star": _star_pts, "hexagon": _hexagon_pts, "pentagon": _pentagon_pts,
    "right_arrow": lambda x, y, w, h: _block_arrow_pts(x, y, w, h, "right_arrow"),
    "left_arrow": lambda x, y, w, h: _block_arrow_pts(x, y, w, h, "left_arrow"),
    "chevron": _chevron_pts,
}


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------
# MSO dash styles as [on, off, ...] segment lengths in multiples of the line
# width (matches PowerPoint's DASH / ROUND_DOT / DASH_DOT).
_DASH_PATTERNS = {
    "dash": [4, 3],
    "dot": [1, 3],
    "dashdot": [4, 3, 1, 3],
}


def _styled_line(draw, p1, p2, color, width, dash):
    """Draw a straight line, solid or dashed per ``dash``."""
    if not dash or dash == "solid" or dash not in _DASH_PATTERNS:
        draw.line([p1, p2], fill=color, width=width)
        return
    pattern = [max(1.0, seg * width) for seg in _DASH_PATTERNS[dash]]
    (x1, y1), (x2, y2) = p1, p2
    total = math.hypot(x2 - x1, y2 - y1)
    if total <= 0:
        return
    ux, uy = (x2 - x1) / total, (y2 - y1) / total
    pos, idx, on = 0.0, 0, True
    while pos < total:
        seg = pattern[idx % len(pattern)]
        end = min(pos + seg, total)
        if on:
            draw.line([(x1 + ux * pos, y1 + uy * pos), (x1 + ux * end, y1 + uy * end)],
                      fill=color, width=width)
        pos, idx, on = end, idx + 1, not on


def _draw_line(draw, theme, el, scale):
    x1, y1 = el["x1"] * scale, el["y1"] * scale
    x2, y2 = el["x2"] * scale, el["y2"] * scale
    color = _rgb(theme, el.get("color"), (60, 60, 60))
    width = max(1, int((el.get("w") or 1) * scale))
    _styled_line(draw, (x1, y1), (x2, y2), color, width, el.get("dash"))
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
            _draw_text_frame(draw, theme, {**frame, "class": spec.get("class") or text_class},
                             (cx, cy, cell_w, row_h), scale)
            cx += cell_w
        cy += row_h
    # No grid lines: the .pptx builder neutralises the table style and relies on
    # header fill + banding only, so a borderless render matches the download.


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def _cover_resize(src, W, H):
    """Resize + centre-crop ``src`` to exactly cover ``W×H`` (preserves mode)."""
    sw, sh = src.size
    if not sw or not sh:
        return src.resize((max(1, W), max(1, H)))
    s = max(W / sw, H / sh)
    nw, nh = max(1, int(sw * s)), max(1, int(sh * s))
    resized = src.resize((nw, nh))
    left, top = (nw - W) // 2, (nh - H) // 2
    return resized.crop((left, top, left + W, top + H))


def _apply_opacity(im, opacity):
    """Fade an RGBA image by scaling its alpha channel (None/1.0 = unchanged)."""
    if opacity is None or opacity >= 1:
        return im
    op = max(0.0, min(1.0, float(opacity)))
    im = im.convert("RGBA")
    im.putalpha(im.split()[3].point(lambda a: int(a * op)))
    return im


def _draw_image(img, theme, el, scale, resolver):
    x, y, w, h = int(el["x"] * scale), int(el["y"] * scale), int(el["w"] * scale), int(el["h"] * scale)
    token = el.get("token", "")
    data = resolver(token) if (resolver and token) else None
    if not data:
        _draw_placeholder(img, x, y, w, h, token=token, scale=scale, theme=theme)
        return
    try:
        src = Image.open(BytesIO(data[0])).convert("RGBA")
    except Exception:  # noqa: BLE001
        _draw_placeholder(img, x, y, w, h, token=token, scale=scale, theme=theme)
        return
    sw, sh = src.size
    if not sw or not sh:
        return
    fit = el.get("fit", "contain")
    if fit == "stretch":
        placed, px, py = src.resize((w, h)), x, y
    elif fit == "cover":
        placed, px, py = _cover_resize(src, w, h), x, y
    else:  # contain
        s = min(w / sw, h / sh)
        nw, nh = max(1, int(sw * s)), max(1, int(sh * s))
        placed, px, py = src.resize((nw, nh)), x + (w - nw) // 2, y + (h - nh) // 2
    placed = _apply_opacity(placed, el.get("opacity"))
    img.paste(placed, (px, py), placed)


def _draw_placeholder(img, x, y, w, h, *, token, scale, theme):
    """A tidy 'image goes here' slot — soft box + image glyph + a caption derived
    from the reservation token, so a reserved mockup/logo reads as intentional
    rather than broken."""
    from chat.slides import icons
    from chat.slides.schema import image_placeholder_caption

    d = ImageDraw.Draw(img)
    d.rounded_rectangle([x, y, x + w, y + h], radius=8 * scale,
                        fill=(236, 239, 233), outline=(199, 207, 196), width=max(1, int(scale)))
    label = image_placeholder_caption(token)
    ink = (138, 148, 139)
    font = _font("Carlito", 12 * scale, False, False)
    # A small image glyph centred above the caption (skip if the box is tiny).
    gs = min(w, h) * 0.34
    has_glyph = gs >= 14 * scale and h >= 40 * scale
    tw = font.getlength(label)
    cap_y = y + h / 2 + (gs * 0.25 if has_glyph else -7 * scale)
    if has_glyph:
        gx, gy = x + (w - gs) / 2, y + h / 2 - gs * 0.72
        icons.draw_icon(d, "image", (gx, gy, gx + gs, gy + gs), ink, max(1, int(1.4 * scale)))
    if tw < w - 8 * scale:
        d.text((x + (w - tw) / 2, cap_y), label, font=font, fill=ink)


# ---------------------------------------------------------------------------
# Charts (preview — the downloaded .pptx carries a native, editable chart)
# ---------------------------------------------------------------------------
def _nice_ticks(vmax, count=4):
    """A 'nice' axis maximum >= vmax and its tick step."""
    if vmax <= 0:
        return 1.0, 0.25
    raw = vmax / count
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    else:
        step = 10 * mag
    return math.ceil(vmax / step) * step, step


def _draw_chart(draw, theme, el, scale, bg=None):
    kind = el.get("chart", "column")
    x, y, w, h = el["x"] * scale, el["y"] * scale, el["w"] * scale, el["h"] * scale
    series = el.get("series") or []
    if not series:
        return
    ramp_names = el.get("colors") or theme.get("colors", {}).get("chart_ramp") or ["accent1"]
    ramp = [_rgb(theme, c, (150, 150, 150)) for c in ramp_names]
    n_vals = max((len(s.get("values") or []) for s in series), default=0)
    cats = el.get("categories") or [str(i + 1) for i in range(n_vals)]
    txt = _rgb(theme, "dk1", (30, 30, 30))
    axis = _rgb(theme, "accent3", (180, 180, 180))
    grid = tuple(min(255, c + 40) for c in axis)

    def fnt(px):
        return _font(theme_mod.font_family(theme, "data"), px * scale, False, False)

    pad = 8 * scale
    top = y + pad
    if el.get("title"):
        f = fnt(12)
        tw = f.getlength(el["title"])
        draw.text((x + (w - tw) / 2, top), el["title"], font=f, fill=txt)
        top += 22 * scale
    show_legend = (kind in ("pie", "doughnut", "waterfall") or len(series) > 1) and el.get("legend", True)
    legend_h = 22 * scale if show_legend else 0
    stacked = bool(el.get("stacked"))
    # Waterfall uses conventional semantic colours (green up / red down / dark total).
    wf_colors = {
        "up": _rgb(theme, "success", (60, 140, 90)),
        "down": _rgb(theme, "danger", (180, 60, 50)),
        "total": _rgb(theme, "dk2", (50, 50, 50)),
    }

    pt_names = el.get("point_colors")
    pt_colors = [_rgb(theme, c, ramp[0]) for c in pt_names] if pt_names else None

    plot = (x + pad, top, w - 2 * pad, y + h - pad - legend_h - top)  # x,y,w,h
    if kind == "pie":
        _chart_pie(draw, plot, series[0], cats, ramp, fnt, txt, el.get("value_labels"))
    elif kind == "doughnut":
        hole = el.get("hole")
        hole = 0.55 if hole is None else max(0.2, min(0.85, float(hole)))
        hole_bg = bg if bg is not None else _rgb(theme, "lt1", (255, 255, 255))
        _chart_pie(draw, plot, series[0], cats, ramp, fnt, txt, el.get("value_labels"),
                   hole=hole, center_label=el.get("center_label", ""), hole_bg=hole_bg, scale=scale)
    elif kind == "waterfall":
        _chart_waterfall(draw, plot, series[0], cats, el.get("totals") or [], wf_colors, fnt, txt, axis, grid, el.get("value_labels"))
    elif kind == "marimekko":
        _chart_marimekko(draw, plot, series, cats, el.get("widths"), ramp, fnt, txt, axis, grid, el.get("value_labels"))
    elif kind == "funnel":
        _chart_funnel(draw, plot, series[0], cats, ramp, fnt, txt, el.get("value_labels"))
    elif kind == "combo":
        _chart_combo(draw, plot, series, cats, ramp, fnt, txt, grid, scale)
    elif kind == "bar":
        _chart_bars(draw, plot, series, cats, ramp, fnt, txt, axis, grid, True, el.get("value_labels"), stacked, pt_colors)
    elif kind in ("line", "area"):
        _chart_lines(draw, plot, series, cats, ramp, fnt, txt, axis, grid, kind == "area", scale)
    else:
        _chart_bars(draw, plot, series, cats, ramp, fnt, txt, axis, grid, False, el.get("value_labels"), stacked, pt_colors)

    if show_legend:
        if kind == "waterfall":
            names = ["Increase", "Decrease", "Total"]
            leg_ramp = [wf_colors["up"], wf_colors["down"], wf_colors["total"]]
        elif kind in ("pie", "doughnut"):
            names, leg_ramp = cats, ramp
        else:
            names, leg_ramp = [s.get("name", "") for s in series], ramp
        _chart_legend(draw, (x + pad, y + h - pad - legend_h + 4, w - 2 * pad, legend_h), names, leg_ramp, fnt(9), txt, scale)


def _val_axis(series):
    vals = [v for s in series for v in (s.get("values") or []) if isinstance(v, (int, float))]
    vmax = max(vals) if vals else 1.0
    vmin = min(vals + [0])
    top, _step = _nice_ticks(vmax if vmax > 0 else 1.0)
    return vmin if vmin < 0 else 0.0, top


def _stacked_axis(series, n_cat):
    """(lo, hi) for a stacked bar chart — hi is a nice tick above the largest
    per-category positive sum (stacking treats negatives as 0)."""
    tot_max = 0.0
    for ci in range(n_cat):
        tot = 0.0
        for s in series:
            vals = s.get("values") or []
            v = vals[ci] if ci < len(vals) else 0
            if isinstance(v, (int, float)) and v > 0:
                tot += v
        tot_max = max(tot_max, tot)
    top, _step = _nice_ticks(tot_max if tot_max > 0 else 1.0)
    return 0.0, top


def _chart_bars(draw, plot, series, cats, ramp, fnt, txt, axis, grid, horizontal, value_labels, stacked=False, point_colors=None):
    px, py, pw, ph = plot
    n_cat, n_ser = len(cats), len(series)
    if n_cat == 0 or n_ser == 0:
        return
    lo, hi = _stacked_axis(series, n_cat) if stacked else _val_axis(series)
    span = (hi - lo) or 1.0
    lbl = fnt(9)
    _, step = _nice_ticks(hi if hi > 0 else 1.0)
    val_gutter = 26   # value-axis labels
    cat_gutter = 30   # category-axis labels

    if horizontal:
        ax, ay, aw, ah = px + cat_gutter, py, pw - cat_gutter, ph - 16
    else:
        ax, ay, aw, ah = px + val_gutter, py, pw - val_gutter, ph - 16

    # value gridlines + labels
    t = lo
    while t <= hi + 1e-9:
        frac = (t - lo) / span
        if horizontal:
            gx = ax + frac * aw
            draw.line([(gx, ay), (gx, ay + ah)], fill=grid, width=1)
            lt = _fmt_num(t)
            draw.text((gx - lbl.getlength(lt) / 2, ay + ah + 3), lt, font=lbl, fill=txt)
        else:
            gy = ay + ah - frac * ah
            draw.line([(ax, gy), (ax + aw, gy)], fill=grid, width=1)
            draw.text((px, gy - 6), _fmt_num(t), font=lbl, fill=txt)
        t += step

    for ci in range(n_cat):
        # PowerPoint bar charts put the first category at the bottom.
        row = (n_cat - 1 - ci) if horizontal else ci
        slot = (ah if horizontal else aw) / n_cat
        if stacked:
            # One full bar per category; series segments stack end to end.
            thick = slot * 0.7
            base = slot * 0.15
            cum = 0.0
            for si, s in enumerate(series):
                vals = s.get("values") or []
                v = vals[ci] if ci < len(vals) else 0
                if not isinstance(v, (int, float)) or v <= 0:
                    continue
                f0, f1 = (cum - lo) / span, (cum + v - lo) / span
                color = ramp[si % len(ramp)]
                if horizontal:
                    by = ay + row * slot + base
                    draw.rectangle([ax + f0 * aw, by, ax + f1 * aw, by + thick], fill=color)
                else:
                    bx = ax + row * slot + base
                    draw.rectangle([bx, ay + ah - f1 * ah, bx + thick, ay + ah - f0 * ah], fill=color)
                cum += v
        else:
            for si, s in enumerate(series):
                vals = s.get("values") or []
                v = vals[ci] if ci < len(vals) else 0
                frac = (v - lo) / span
                # Single-series bar highlighting: colour each bar by category.
                if point_colors and n_ser == 1 and ci < len(point_colors):
                    color = point_colors[ci]
                else:
                    color = ramp[si % len(ramp)]
                thick = slot * 0.7 / n_ser
                if horizontal:
                    by = ay + row * slot + slot * 0.15 + si * thick
                    draw.rectangle([ax, by, ax + frac * aw, by + thick], fill=color)
                else:
                    bx = ax + row * slot + slot * 0.15 + si * thick
                    draw.rectangle([bx, ay + ah - frac * ah, bx + thick, ay + ah], fill=color)
        # category label
        f = fnt(9)
        if horizontal:
            cw = f.getlength(cats[ci])
            draw.text((px + cat_gutter - cw - 4, ay + row * slot + slot / 2 - 6), cats[ci], font=f, fill=txt)
        else:
            cw = f.getlength(cats[ci])
            draw.text((ax + row * slot + slot / 2 - cw / 2, ay + ah + 3), cats[ci], font=f, fill=txt)


def _waterfall_bars(values, totals):
    """Compute per-step ``(low, high, role, delta)`` + the running total after each
    step for a waterfall/bridge. ``totals`` (indices, or empty=first) are absolute
    bars from zero; the rest are deltas that float on the running total."""
    tset = set(totals) if totals else {0}
    running = 0.0
    bars, edges = [], []
    for i, val in enumerate(values):
        val = float(val) if isinstance(val, (int, float)) else 0.0
        if i in tset:
            lo_i, hi_i, role = 0.0, val, "total"
            running = val
        else:
            v0, v1 = running, running + val
            lo_i, hi_i, role = min(v0, v1), max(v0, v1), ("up" if val >= 0 else "down")
            running = v1
        bars.append((lo_i, hi_i, role, val))
        edges.append(running)
    return bars, edges


def _chart_waterfall(draw, plot, series, cats, totals, colors, fnt, txt, axis, grid, value_labels):
    px, py, pw, ph = plot
    values = list(series.get("values") or [])
    n = len(values)
    if n == 0:
        return
    bars, edges = _waterfall_bars(values, totals)
    all_v = [0.0] + [b[0] for b in bars] + [b[1] for b in bars]
    vmin, vmax = min(all_v), max(all_v)
    top, step = _nice_ticks(vmax if vmax > 0 else 1.0)
    lo, hi = (vmin if vmin < 0 else 0.0), top
    span = (hi - lo) or 1.0
    lbl = fnt(9)
    val_gutter = 26
    ax, ay, aw, ah = px + val_gutter, py, pw - val_gutter, ph - 16

    def to_y(v):
        return ay + ah - (v - lo) / span * ah

    # value gridlines + labels
    t = lo
    while t <= hi + 1e-9:
        gy = to_y(t)
        draw.line([(ax, gy), (ax + aw, gy)], fill=grid, width=1)
        draw.text((px, gy - 6), _fmt_num(t), font=lbl, fill=txt)
        t += step

    slot = aw / n
    col_w = slot * 0.6
    col_off = (slot - col_w) / 2
    prev_right = prev_level_y = None
    for i, (lo_i, hi_i, role, delta) in enumerate(bars):
        bx = ax + i * slot + col_off
        y_hi, y_lo = to_y(hi_i), to_y(lo_i)
        draw.rectangle([bx, y_hi, bx + col_w, y_lo], fill=colors[role])
        # connector from the previous bar's running level to this bar's left edge
        if prev_right is not None:
            draw.line([(prev_right, prev_level_y), (bx, prev_level_y)], fill=axis, width=1)
        prev_right, prev_level_y = bx + col_w, to_y(edges[i])
        # category label
        clbl = cats[i] if i < len(cats) else ""
        cw = lbl.getlength(clbl)
        draw.text((ax + i * slot + slot / 2 - cw / 2, ay + ah + 3), clbl, font=lbl, fill=txt)
        # value label above the bar
        if value_labels:
            vs = _fmt_num(hi_i) if role == "total" else ("+" if delta >= 0 else "−") + _fmt_num(abs(delta))
            vw = lbl.getlength(vs)
            draw.text((bx + col_w / 2 - vw / 2, y_hi - 12), vs, font=lbl, fill=txt)


def _marimekko_columns(series, cats, widths):
    """Per-column ``(width, total)`` for a Marimekko: width = the size dimension
    (explicit ``widths`` or the column's own total), total = sum of the stack."""
    n_cat = len(cats)
    totals = []
    for ci in range(n_cat):
        t = 0.0
        for s in series:
            vals = s.get("values") or []
            v = vals[ci] if ci < len(vals) else 0
            if isinstance(v, (int, float)) and v > 0:
                t += v
        totals.append(t)
    if widths and len(widths) >= n_cat:
        ws = [max(0.0, float(widths[ci])) for ci in range(n_cat)]
    else:
        ws = list(totals)
    return ws, totals


def _chart_marimekko(draw, plot, series, cats, widths, ramp, fnt, txt, axis, grid, value_labels):
    px, py, pw, ph = plot
    n_cat = len(cats)
    if n_cat == 0 or not series:
        return
    ws, totals = _marimekko_columns(series, cats, widths)
    w_sum = sum(ws) or 1.0
    lbl = fnt(9)
    val_gutter, cat_gutter = 34, 26
    gap = 2  # px between columns
    ax, ay, aw, ah = px + val_gutter, py, pw - val_gutter, ph - cat_gutter

    # 0/25/50/75/100% gridlines + labels (share axis, right-aligned in the gutter)
    for pct in (0, 25, 50, 75, 100):
        gy = ay + ah - (pct / 100.0) * ah
        draw.line([(ax, gy), (ax + aw, gy)], fill=grid, width=1)
        pl = f"{pct}%"
        draw.text((ax - 4 - lbl.getlength(pl), gy - 6), pl, font=lbl, fill=txt)

    usable = aw - gap * (n_cat - 1)
    cx = ax
    for ci in range(n_cat):
        col_w = usable * (ws[ci] / w_sum)
        total = totals[ci] or 1.0
        y_bot = ay + ah
        for si, s in enumerate(series):
            vals = s.get("values") or []
            v = vals[ci] if ci < len(vals) else 0
            if not isinstance(v, (int, float)) or v <= 0:
                continue
            seg_h = (v / total) * ah
            y_top = y_bot - seg_h
            draw.rectangle([cx, y_top, cx + col_w, y_bot], fill=ramp[si % len(ramp)])
            if value_labels and seg_h > 14 and col_w > 22:
                ptxt = f"{v / total * 100:.0f}%"
                tw = lbl.getlength(ptxt)
                draw.text((cx + col_w / 2 - tw / 2, (y_top + y_bot) / 2 - 6), ptxt,
                          font=lbl, fill=(255, 255, 255))
            y_bot = y_top
        # category label (centred under the column)
        clbl = cats[ci] if ci < len(cats) else ""
        cw = lbl.getlength(clbl)
        draw.text((cx + col_w / 2 - cw / 2, ay + ah + 4), clbl, font=lbl, fill=txt)
        cx += col_w + gap


def _combo_axis(group):
    vals = [v for s in group for v in (s.get("values") or []) if isinstance(v, (int, float))]
    vmax = max(vals) if vals else 1.0
    vmin = min(vals + [0])
    top, step = _nice_ticks(vmax if vmax > 0 else 1.0)
    return (vmin if vmin < 0 else 0.0), top, step


def _chart_combo(draw, plot, series, cats, ramp, fnt, txt, grid, scale):
    """A bar+line combo with an optional secondary (right) value axis — e.g.
    revenue bars on the left axis + margin % line on the right axis."""
    px, py, pw, ph = plot
    n_cat = len(cats)
    if n_cat == 0 or not series:
        return
    lbl = fnt(9)
    has_sec = any(s.get("axis") == "secondary" for s in series)
    prim = [s for s in series if s.get("axis") != "secondary"]
    sec = [s for s in series if s.get("axis") == "secondary"]
    plo, phi, pstep = _combo_axis(prim) if prim else (0.0, 1.0, 1.0)
    slo, shi, sstep = _combo_axis(sec) if sec else (0.0, 1.0, 1.0)
    pspan, sspan = (phi - plo) or 1.0, (shi - slo) or 1.0
    lgut, rgut = 30, (36 if has_sec else 4)
    ax, ay, aw, ah = px + lgut, py, pw - lgut - rgut, ph - 16

    def yfor(v, secondary):
        lo, hi, span = (slo, shi, sspan) if secondary else (plo, phi, pspan)
        return ay + ah - ((v - lo) / span) * ah

    # left axis gridlines + labels
    t = plo
    while t <= phi + 1e-9:
        gy = ay + ah - (t - plo) / pspan * ah
        draw.line([(ax, gy), (ax + aw, gy)], fill=grid, width=1)
        draw.text((px, gy - 6), _fmt_num(t), font=lbl, fill=txt)
        t += pstep
    # right axis labels
    if has_sec:
        t = slo
        while t <= shi + 1e-9:
            gy = ay + ah - (t - slo) / sspan * ah
            draw.text((ax + aw + 5, gy - 6), _fmt_num(t), font=lbl, fill=txt)
            t += sstep

    bar_series = [s for s in series if s.get("kind", "bar") == "bar"]
    line_series = [s for s in series if s.get("kind") == "line"]
    slot = aw / n_cat
    nb = max(1, len(bar_series))
    for ci in range(n_cat):
        for bi, s in enumerate(bar_series):
            secondary = s.get("axis") == "secondary"
            vals = s.get("values") or []
            v = vals[ci] if ci < len(vals) else 0
            thick = slot * 0.7 / nb
            bx = ax + ci * slot + slot * 0.15 + bi * thick
            draw.rectangle([bx, yfor(v, secondary), bx + thick, ay + ah],
                           fill=ramp[series.index(s) % len(ramp)])
        clbl = cats[ci] if ci < len(cats) else ""
        cw = lbl.getlength(clbl)
        draw.text((ax + ci * slot + slot / 2 - cw / 2, ay + ah + 3), clbl, font=lbl, fill=txt)

    for s in line_series:
        secondary = s.get("axis") == "secondary"
        vals = s.get("values") or []
        pts = [(ax + ci * slot + slot / 2, yfor(vals[ci] if ci < len(vals) else 0, secondary))
               for ci in range(n_cat)]
        color = ramp[series.index(s) % len(ramp)]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=max(2, int(2 * scale)), joint="curve")
        r = max(2, int(2.5 * scale))
        for x0, y0 in pts:
            draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=color)


def render_chart_png(theme, el, k: float = 3.0) -> bytes:
    """Render a single chart element to transparent PNG bytes (for embedding a
    chart type the .pptx can't build natively — e.g. combo — as a picture)."""
    w_px = max(1, int(el.get("w", 480) * k))
    h_px = max(1, int(el.get("h", 300) * k))
    img = Image.new("RGBA", (w_px, h_px), (0, 0, 0, 0))
    local = dict(el)
    local["x"], local["y"] = 0.0, 0.0
    _draw_chart(ImageDraw.Draw(img), theme, local, k, bg=None)
    buf = BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _chart_funnel(draw, plot, series, cats, ramp, fnt, txt, value_labels):
    px, py, pw, ph = plot
    vals = [max(0.0, float(v)) for v in (series.get("values") or []) if isinstance(v, (int, float))]
    n = len(vals)
    if n == 0:
        return
    vmax = max(vals) or 1.0
    cx = px + pw / 2
    gap = 4
    band = (ph - gap * (n - 1)) / n
    lbl, small = fnt(10), fnt(9)
    for i, v in enumerate(vals):
        y0 = py + i * (band + gap)
        y1 = y0 + band
        wd = max(10.0, (v / vmax) * pw * 0.9)
        draw.rectangle([cx - wd / 2, y0, cx + wd / 2, y1], fill=ramp[i % len(ramp)])
        name = cats[i] if i < len(cats) else ""
        label = (f"{name}  {_fmt_num(v)}").strip() if name else _fmt_num(v)
        tw = lbl.getlength(label)
        if tw < wd - 8:
            draw.text((cx - tw / 2, (y0 + y1) / 2 - 7), label, font=lbl, fill=(255, 255, 255))
        else:  # doesn't fit inside the band -> name to the left
            draw.text((px, (y0 + y1) / 2 - 6), name or _fmt_num(v), font=small, fill=txt)
        # stage-to-stage conversion %, to the right
        if value_labels and i > 0 and vals[i - 1] > 0:
            pct = f"{v / vals[i - 1] * 100:.0f}%"
            draw.text((cx + wd / 2 + 8, (y0 + y1) / 2 - 6), pct, font=small, fill=txt)


def _chart_lines(draw, plot, series, cats, ramp, fnt, txt, axis, grid, area, scale):
    px, py, pw, ph = plot
    lo, hi = _val_axis(series)
    span = (hi - lo) or 1.0
    lab_gutter = 34
    ax, ay, aw, ah = px + lab_gutter, py, pw - lab_gutter, ph - 16
    _, step = _nice_ticks(hi if hi > 0 else 1.0)
    lbl = fnt(9)
    t = lo
    while t <= hi + 1e-9:
        gy = ay + ah - (t - lo) / span * ah
        draw.line([(ax, gy), (ax + aw, gy)], fill=grid, width=1)
        draw.text((px, gy - 6), _fmt_num(t), font=lbl, fill=txt)
        t += step
    n = len(cats)
    if n == 0:
        return
    step_x = aw / max(1, n - 1) if n > 1 else aw
    for si, s in enumerate(series):
        vals = s.get("values") or []
        color = ramp[si % len(ramp)]
        pts = []
        for ci in range(n):
            v = vals[ci] if ci < len(vals) else 0
            gx = ax + (ci * step_x if n > 1 else aw / 2)
            gy = ay + ah - (v - lo) / span * ah
            pts.append((gx, gy))
        if area and len(pts) > 1:
            poly = pts + [(pts[-1][0], ay + ah), (pts[0][0], ay + ah)]
            draw.polygon(poly, fill=color + (90,) if len(color) == 4 else (*color, 90))
        if len(pts) > 1:
            draw.line(pts, fill=color, width=max(2, int(2 * scale)), joint="curve")
        r = max(2, int(2.5 * scale))
        for gx, gy in pts:
            draw.ellipse([gx - r, gy - r, gx + r, gy + r], fill=color)
    for ci in range(n):
        f = fnt(9)
        cw = f.getlength(cats[ci])
        gx = ax + (ci * step_x if n > 1 else aw / 2)
        draw.text((gx - cw / 2, ay + ah + 3), cats[ci], font=f, fill=txt)


def _chart_pie(draw, plot, series, cats, ramp, fnt, txt, value_labels,
               hole=0.0, center_label="", hole_bg=None, scale=1.0):
    px, py, pw, ph = plot
    vals = [max(0.0, v) for v in (series.get("values") or [])]
    total = sum(vals) or 1.0
    d = min(pw, ph) * 0.9
    cx, cy = px + pw / 2, py + ph / 2
    box = [cx - d / 2, cy - d / 2, cx + d / 2, cy + d / 2]
    start = -90.0
    f = fnt(9)
    # A doughnut labels each slice further out (the hole eats the centre).
    lab_r = (d / 2) * (0.5 + hole / 2.5) if hole else (d / 2) * 0.62
    for i, v in enumerate(vals):
        sweep = v / total * 360.0
        draw.pieslice(box, start, start + sweep, fill=ramp[i % len(ramp)])
        if value_labels and v > 0:
            mid = math.radians(start + sweep / 2)
            lx = cx + lab_r * math.cos(mid)
            ly = cy + lab_r * math.sin(mid)
            pct = f"{v / total * 100:.0f}%"
            tw = f.getlength(pct)
            draw.text((lx - tw / 2, ly - 6), pct, font=f, fill=(255, 255, 255))
        start += sweep
    if hole and hole > 0:
        hr = (d / 2) * hole
        draw.ellipse([cx - hr, cy - hr, cx + hr, cy + hr], fill=hole_bg or (255, 255, 255))
        if center_label:
            cf = fnt(max(10.0, (hr * 2) / scale * 0.34))
            cw = cf.getlength(center_label)
            draw.text((cx - cw / 2, cy - cf.size / 2), center_label, font=cf, fill=txt)


def _chart_legend(draw, box, names, ramp, font, txt, scale):
    bx, by, bw, bh = box
    sw = 10 * scale
    gap = 14 * scale
    items = [(n, ramp[i % len(ramp)]) for i, n in enumerate(names)]
    widths = [sw + 4 * scale + font.getlength(n) for n, _ in items]
    total = sum(widths) + gap * (len(items) - 1)
    x = bx + max(0, (bw - total) / 2)
    for (name, color), wdt in zip(items, widths):
        draw.rectangle([x, by + bh / 2 - sw / 2, x + sw, by + bh / 2 + sw / 2], fill=color)
        draw.text((x + sw + 4 * scale, by + bh / 2 - font.size / 2), name, font=font, fill=txt)
        x += wdt + gap


def _fmt_num(v):
    if abs(v) >= 1000:
        return f"{v/1000:.0f}k" if v % 1000 == 0 else f"{v/1000:.1f}k"
    if v == int(v):
        return str(int(v))
    return f"{v:.1f}"


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
# Element dispatch (+ rotation)
# ---------------------------------------------------------------------------
def _draw_element(draw, img, theme, el, scale, resolver, bg=None):
    etype = el.get("type")
    if etype == "text":
        frame = {"class": el.get("class"), "paragraphs": el.get("paragraphs"),
                 "valign": el.get("valign", "top")}
        box = (el["x"] * scale, el["y"] * scale, el["w"] * scale, el["h"] * scale)
        _draw_text_frame(draw, theme, frame, box, scale)
    elif etype == "shape":
        op = el.get("opacity")
        if op is not None and op < 1:  # translucent panel: draw on a layer, fade, composite
            layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            _draw_shape(ImageDraw.Draw(layer), theme, el, scale)
            layer = _apply_opacity(layer, op)
            img.paste(layer, (0, 0), layer)
        else:
            _draw_shape(draw, theme, el, scale)
    elif etype == "icon":
        op = el.get("opacity")
        if op is not None and op < 1:
            layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            _draw_icon(ImageDraw.Draw(layer), theme, el, scale)
            layer = _apply_opacity(layer, op)
            img.paste(layer, (0, 0), layer)
        else:
            _draw_icon(draw, theme, el, scale)
    elif etype == "image":
        _draw_image(img, theme, el, scale, resolver)
    elif etype == "table":
        _draw_table(draw, theme, el, scale)
    elif etype == "line":
        _draw_line(draw, theme, el, scale)
    elif etype == "chart":
        _draw_chart(draw, theme, el, scale, bg)


def _draw_element_rotated(img, theme, el, scale, resolver, angle):
    """Draw a rotatable element (text/shape/image) onto a transparent square
    layer, rotate it about its centre (MSO rotation is clockwise-positive), and
    composite it back — so the element rotates in place like in PowerPoint."""
    ew, eh = el["w"] * scale, el["h"] * scale
    ex, ey = el["x"] * scale, el["y"] * scale
    side = int(math.hypot(ew, eh)) + 4
    layer = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    ldraw = ImageDraw.Draw(layer)
    offx, offy = (side - ew) / 2, (side - eh) / 2
    local = dict(el)
    local["x"], local["y"] = offx / scale, offy / scale
    local.pop("rotation", None)
    _draw_element(ldraw, layer, theme, local, scale, resolver)
    rotated = layer.rotate(-angle, resample=Image.BICUBIC, expand=True)
    cx, cy = ex + ew / 2, ey + eh / 2
    img.paste(rotated, (int(round(cx - rotated.width / 2)), int(round(cy - rotated.height / 2))), rotated)


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

    # Full-bleed background image (drawn behind everything), then an optional
    # scrim — a translucent colour wash so text stays legible over the photo.
    bgtok = slide.get("bg_image")
    if bgtok and image_resolver:
        data = image_resolver(bgtok)
        if data:
            try:
                img.paste(_cover_resize(Image.open(BytesIO(data[0])).convert("RGB"), W, H), (0, 0))
            except Exception:  # noqa: BLE001
                logger.warning("pillow render: bg image failed", exc_info=True)
    scrim = slide.get("bg_scrim")
    if scrim:
        col = _rgb(theme, scrim.get("color", "dk1"), (0, 0, 0))
        op = max(0.0, min(1.0, float(scrim.get("opacity", 0.4))))
        img = Image.blend(img, Image.new("RGB", (W, H), col), op)

    draw = ImageDraw.Draw(img)

    for el in slide.get("elements") or []:
        etype = el.get("type")
        try:
            rot = el.get("rotation")
            if rot and etype in ("text", "shape", "image", "icon"):
                _draw_element_rotated(img, theme, el, scale, image_resolver, float(rot))
            else:
                _draw_element(draw, img, theme, el, scale, image_resolver, bg)
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
