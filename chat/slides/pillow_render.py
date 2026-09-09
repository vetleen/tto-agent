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

from PIL import Image, ImageDraw, ImageFilter, ImageFont

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
def _opacity(v):
    """Coerce an opacity to a float in [0,1], or None. Tolerates a stringified
    number (lax schema validation can let ``"0.5"`` through) instead of letting
    a later ``"0.5" < 1`` raise TypeError and silently drop the element."""
    if v is None:
        return None
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return None


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


class _FontBook:
    """Per-render loader for org-resolved font faces (bundled or uploaded).

    Built from :func:`chat.slides.font_resolve.resolve_deck_font_faces` output —
    ``{family: {"faces": [core.fonts.FontFace, …]}}`` — and attached to the resolved
    theme as ``theme["_fontbook"]`` so ``_run_font`` can rasterise uploaded fonts from
    bytes. Kept per-render (not a module ``lru_cache``) so two orgs' fonts under the
    same family name never collide.
    """
    def __init__(self, font_faces: dict | None):
        self._faces = font_faces or {}
        self._cache: dict = {}

    @staticmethod
    def _pick(faces, bold, italic):
        want_w = 700 if bold else 400
        want_s = "italic" if italic else "normal"
        return min(faces, key=lambda f: (0 if f.style == want_s else 1, abs(f.weight - want_w)))

    def pil_font(self, family, px, bold, italic):
        rec = self._faces.get(family)
        faces = rec.get("faces") if rec else None
        if not faces:
            return None
        px = max(1, int(round(px)))
        key = (family, bool(bold), bool(italic), px)
        if key in self._cache:
            return self._cache[key]
        font = None
        try:
            face = self._pick(faces, bold, italic)
            font = ImageFont.truetype(BytesIO(face.data), px)
        except Exception:  # noqa: BLE001 — fall back to the bundled disk lookup
            font = None
        self._cache[key] = font
        return font


def _run_font(theme: dict, style: dict, scale: float) -> ImageFont.FreeTypeFont:
    family = theme_mod.font_family(theme, style.get("font"))
    px = (style.get("size") or 14) * scale
    bold, italic = bool(style.get("bold")), bool(style.get("italic"))
    book = theme.get("_fontbook")
    if book is not None:
        f = book.pil_font(family, px, bold, italic)
        if f is not None:
            return f
    return _font(family, px, bold, italic)


# The brand serif/sans (Caladea/Carlito) lack many symbol glyphs the model likes
# to use inline — arrows ↑↓→, triangles ▲▼►, checks ✓ — which would otherwise
# render as .notdef "tofu" boxes. We fall back per-glyph through a chain: Arimo
# (Arial-metric, covers most arrows/shapes), then NotoSansSymbols2 (dingbats and
# ornament brackets like ❯ U+276F that no metric-compatible family carries), then
# Gelasio (has the math angle brackets ⟨⟩ the others lack). PowerPoint
# substitutes similarly for the .pptx.
_SYMBOL_FALLBACKS = ("Arimo", "NotoSansSymbols2", "Gelasio")


@lru_cache(maxsize=32)
def _font_codepoints(family: str) -> frozenset:
    try:
        from fontTools.ttLib import TTFont

        path = _FONTS_DIR / family / f"{family}-Regular.ttf"
        if not path.exists():
            return frozenset()
        return frozenset(TTFont(str(path)).getBestCmap().keys())
    except Exception:  # noqa: BLE001
        return frozenset()


def _fallback_family(cp: int, prim: frozenset) -> str | None:
    """The first fallback family whose cmap has ``cp``, or None when the primary
    font covers it (ASCII assumed covered) or nothing bundled does."""
    if cp < 0x80 or cp in prim:
        return None
    for fam in _SYMBOL_FALLBACKS:
        if cp in _font_codepoints(fam):
            return fam
    return None


def _coverage_segments(text: str, family: str):
    """Split ``text`` into (segment, fallback_family_or_None) runs — a char uses a
    fallback when it's non-ASCII, the primary font lacks it, and a chain font has it."""
    prim = _font_codepoints(family)
    out, cur, cur_fb = [], "", None
    for ch in text:
        fb = _fallback_family(ord(ch), prim)
        if not cur:
            cur, cur_fb = ch, fb
        elif fb == cur_fb:
            cur += ch
        else:
            out.append((cur, cur_fb))
            cur, cur_fb = ch, fb
    if cur:
        out.append((cur, cur_fb))
    return out


# ---------------------------------------------------------------------------
# Text layout
#   A paragraph's runs are tokenised into styled words, greedily packed into
#   lines that fit the box width, then drawn with the paragraph's alignment and
#   the text-frame's vertical anchor. Line height tracks the tallest run.
# ---------------------------------------------------------------------------
class _Word:
    __slots__ = ("text", "font", "color", "underline", "w", "ascent", "descent", "space_w", "segs")

    def __init__(self, text, font, color, underline, family=None, fb_fonts=None):
        self.text = text
        self.font = font
        self.color = color
        self.underline = underline
        # (segment, font) pairs so glyphs the primary font lacks draw with a
        # fallback face; plain text stays a single segment on the primary font.
        if fb_fonts and family:
            self.segs, self.w = [], 0.0
            for seg, fb_fam in _coverage_segments(text, family):
                f = fb_fonts.get(fb_fam, font) if fb_fam else font
                self.segs.append((seg, f))
                self.w += f.getlength(seg)
        else:
            self.segs = [(text, font)]
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
        family = theme_mod.font_family(theme, style.get("font"))
        fb_fonts = None
        if any(ord(c) >= 0x80 for c in text):  # only build fallbacks for non-ASCII runs
            fb_fonts = {
                fam: _font(fam, (style.get("size") or 14) * scale,
                           bool(style.get("bold")), bool(style.get("italic")))
                for fam in _SYMBOL_FALLBACKS
            }
        segments = text.split("\n")
        for si, seg in enumerate(segments):
            for tok in seg.split(" "):
                if tok == "":
                    continue
                words.append(_Word(tok, font, color, underline, family, fb_fonts))
            if si < len(segments) - 1 and words:
                breaks.add(len(words) - 1)  # hard break after the last word so far
    return words, breaks


def _split_word_head(word, max_w):
    """Split a word wider than a line: the longest prefix (≥1 char) that fits
    ``max_w`` and the remainder as a new word (``None`` if nothing is left).
    PowerPoint breaks an over-long word mid-word instead of overflowing the box
    — a label like "down_arrow" in a narrow arrow shaft wraps as dow/n_ar/row —
    so the preview must too, or it hides the overflow the download will show."""
    text = word.text
    cut = 1
    for n in range(2, len(text) + 1):
        if word.font.getlength(text[:n]) > max_w:
            break
        cut = n
    head = _Word(text[:cut], word.font, word.color, word.underline)
    tail = _Word(text[cut:], word.font, word.color, word.underline) if cut < len(text) else None
    return head, tail


def _wrap(words, breaks, max_w):
    """Greedily pack words into lines that fit ``max_w`` px. Returns list of lines;
    each line is a list of ``(word, x_offset)`` plus its width and height."""
    lines = []
    cur, cur_w = [], 0.0
    for i, word in enumerate(words):
        pending = [word]
        while pending:
            word = pending.pop(0)
            add = word.w if not cur else word.space_w + word.w
            if cur and cur_w + add > max_w:
                lines.append(cur)
                cur, cur_w = [], 0.0
                add = word.w
            if not cur and word.w > max_w and len(word.text) > 1:
                word, tail = _split_word_head(word, max_w)
                add = word.w
                if tail is not None:
                    pending.insert(0, tail)
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


def _bullet_font(theme, style, scale, char):
    """The font to draw a bullet glyph with — the paragraph's own face, or the
    first symbol-fallback family that has the glyph (Carlito has no ‣, and no
    metric-compatible family has ornament brackets like ❯)."""
    family = theme_mod.font_family(theme, style.get("font"))

    def covers(fam):
        cps = _font_codepoints(fam)
        return all(ord(c) < 0x80 or ord(c) in cps for c in char)

    if not covers(family):
        for fam in _SYMBOL_FALLBACKS:
            if covers(fam):
                return _font(fam, (style.get("size") or 14) * scale,
                             bool(style.get("bold")), bool(style.get("italic")))
    return _run_font(theme, style, scale)


def _draw_text_frame(draw, theme, frame, box, scale, insets=None):
    """Render a text frame (paragraphs) inside ``box`` (x, y, w, h in px).

    The box is the full element/shape/cell rect; the standard PowerPoint text
    insets are applied here so callers pass unpadded rects. ``insets`` overrides
    them as ``(lr_pt, tb_pt)`` (panel shapes pass roomier ones).
    """
    bx, by, bw, bh = box
    ins_lr = (insets[0] if insets else _INSET_LR) * scale
    ins_tb = (insets[1] if insets else _INSET_TB) * scale
    bx, by = bx + ins_lr, by + ins_tb
    bw, bh = max(1.0, bw - 2 * ins_lr), max(1.0, bh - 2 * ins_tb)
    default_cls = frame.get("class")
    paragraphs = frame.get("paragraphs") or []
    valign = frame.get("valign", "top")

    # First pass: lay out every paragraph into positioned lines to get total height.
    laid = []  # list of dicts: {lines, align, indent, bullet, para_gap, line_h}
    total_h = 0.0
    for para in paragraphs:
        base_style = theme_mod.base_text_style(theme, para.get("class") or default_cls)
        words, breaks = _para_words(theme, para, base_style, scale)
        level = int(para.get("level", 0) or 0)
        is_bullet = bool(para.get("bullet"))
        # Bullet: PowerPoint hangs the text at a fixed indent (0.25") from the
        # bullet glyph, which sits at the (level-nested) left edge. Match that so
        # wrapped lines align under the text, not the bullet. The glyph itself
        # steps through the theme's per-level chars (‣ / – / ◦ by default).
        level_indent = level * 20 * scale
        hang = 18 * scale
        indent = level_indent
        bullet_glyph = None
        if is_bullet:
            bchar = theme_mod.bullet_char_for_level(theme, level)
            bfont = _bullet_font(theme, base_style, scale, bchar)
            bullet_glyph = (bchar, bfont, _rgb(theme, base_style.get("color"), (0, 0, 0)), level_indent)
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
                sx = x
                for seg, f in word.segs:
                    draw.text((sx, cursor_y), seg, font=f, fill=word.color)
                    sx += f.getlength(seg)
                if word.underline:
                    uy = cursor_y + word.ascent + max(1, int(scale))
                    draw.line([(x, uy), (x + word.w, uy)], fill=word.color, width=max(1, int(scale)))
                x += word.w
            cursor_y += lh
        cursor_y += p["gap"]


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------
def _gradient_rgb(w, h, c1, c2, angle=90.0):
    """A ``w×h`` RGB image with a two-colour linear gradient. ``angle`` is degrees:
    0 = left→right, 90 = top→bottom (matches OOXML ``gradient_angle``)."""
    import numpy as np

    w, h = max(1, int(round(w))), max(1, int(round(h)))
    a = math.radians(angle)
    dx, dy = math.cos(a), math.sin(a)
    gx, gy = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    proj = gx * dx + gy * dy
    lo, hi = float(proj.min()), float(proj.max())
    t = (proj - lo) / (hi - lo) if hi > lo else np.zeros_like(proj)
    c1 = np.array(c1, dtype=float)
    c2 = np.array(c2, dtype=float)
    arr = c1[None, None, :] * (1.0 - t[:, :, None]) + c2[None, None, :] * t[:, :, None]
    return Image.fromarray(arr.astype("uint8"), "RGB")


def _gradient_colors(theme, grad):
    """Resolve a gradient dict's ``from``/``to`` to RGB tuples."""
    c1 = _rgb(theme, grad.get("from", "dk2"), (120, 120, 120))
    c2 = _rgb(theme, grad.get("to", "dk1"), (40, 40, 40))
    return c1, c2


def round_rect_radius_pt(w_pt, h_pt) -> float:
    """Corner radius (pt) for a rounded_rect: 14% of the short side, capped at
    12pt so a big content panel gets a subtle card corner rather than a pill
    curve. The .pptx builder applies the same fraction via the shape adjustment
    so the download matches the preview."""
    return min(0.14 * min(w_pt, h_pt), 12.0)


def panel_text_insets(shape_name: str, w_pt: float, h_pt: float) -> tuple[float, float]:
    """Text insets ``(lr_pt, tb_pt)`` for a shape's text frame. Plain and rounded
    rectangles are the "content panel" shapes — their text gets more breathing
    room than PowerPoint's cramped 7.2/3.6pt defaults (scaled down for small
    panels); every other shape keeps the defaults so small ovals/chevrons don't
    lose wrap width. Shared with the .pptx builder for preview/download parity."""
    if shape_name in ("rect", "rectangle", "rounded_rect"):
        return (max(_INSET_LR, min(16.0, 0.08 * w_pt)),
                max(_INSET_TB, min(12.0, 0.08 * h_pt)))
    return (_INSET_LR, _INSET_TB)


def shape_text_rect(shape_name: str, x: float, y: float, w: float, h: float):
    """The preset's inner TEXT rectangle ``(x, y, w, h)`` — where PowerPoint lays
    a shape's text out (the OOXML ``<a:rect>`` of each preset at its default
    adjustments): a diamond's is the inscribed half-size box, an arrow's its
    shaft, a chevron's excludes the notches. The preview wraps text inside this
    rect so it breaks lines exactly where the download does. Any units."""
    ss = min(w, h)
    if shape_name in ("oval", "ellipse", "circle"):
        ix, iy = w * 0.14645, h * 0.14645  # (1 − cos 45°) / 2
        return (x + ix, y + iy, w - 2 * ix, h - 2 * iy)
    if shape_name == "diamond":
        return (x + w / 4, y + h / 4, w / 2, h / 2)
    if shape_name == "chevron":
        n = ss * 0.5
        return (x + n, y, max(1.0, w - 2 * n), h)
    if shape_name == "pentagon":
        return (x, y, w - ss * 0.25, h)
    if shape_name == "hexagon":
        # OOXML hexagon il/it: a piecewise fraction of w/h that shrinks the
        # inset as the box gets wider than tall (q8/24 with q8 ∈ [1, 4]).
        max_adj = 50000 * w / ss
        a = min(25000, max_adj)
        q1 = -max_adj / 2
        q2 = a + q1
        q3, q4 = (4, 3) if q2 > 0 else (2, 2)
        q5 = q1 if q2 > 0 else 0
        q6 = (a + q5) / q1
        q8 = q3 - q6 * q4
        ix, iy = w * q8 / 24, h * q8 / 24
        return (x + ix, y + iy, w - 2 * ix, h - 2 * iy)
    if shape_name == "star":
        # The inner pentagon's bounding box (see _star_pts for the geometry).
        cx, cy = x + w / 2, y + (h / 2) * _STAR_VF
        iw, ih = (w / 2) * _STAR_HF * _STAR_INNER, (h / 2) * _STAR_VF * _STAR_INNER
        left, top = cx - iw * math.cos(math.radians(18)), cy - ih * math.sin(math.radians(54))
        return (left, top, 2 * iw * math.cos(math.radians(18)), ih * (1 + math.sin(math.radians(54))))
    if shape_name in ("right_arrow", "left_arrow"):
        head = ss * 0.5
        tx = x + (head / 2 if shape_name == "left_arrow" else 0)
        return (tx, y + h / 4, w - head / 2, h / 2)
    if shape_name in ("up_arrow", "down_arrow"):
        head = ss * 0.5
        ty = y + (head / 2 if shape_name == "up_arrow" else 0)
        return (x + w / 4, ty, w / 2, h - head / 2)
    if shape_name == "plus":  # mathPlus: the horizontal arm
        return (x + w * (0.5 - 0.36745), y + h / 2 - ss * 0.1176, w * 2 * 0.36745, ss * 2 * 0.1176)
    if shape_name == "cross":  # plus: the centre square
        i = ss * 0.25
        return (x + i, y + i, w - 2 * i, h - 2 * i)
    return (x, y, w, h)  # rect / rounded_rect / anything else: the full box


def _fill_shape_mask(md, name, el, ox, oy, w, h, scale):
    """Paint the silhouette of shape ``name`` (box ``w``×``h`` px at ``(ox, oy)``)
    at full alpha onto an "L" mask via ``md``."""
    if name in ("oval", "ellipse", "circle"):
        md.ellipse([ox, oy, ox + w - 1, oy + h - 1], fill=255)
    elif name == "rounded_rect":
        md.rounded_rectangle([ox, oy, ox + w - 1, oy + h - 1],
                             radius=round_rect_radius_pt(el["w"], el["h"]) * scale, fill=255)
    elif name in _POLY_SHAPES:
        md.polygon(_POLY_SHAPES[name](ox, oy, w, h), fill=255)
    else:
        md.rectangle([ox, oy, ox + w - 1, oy + h - 1], fill=255)


def _draw_shape_shadow(img, name, el, x, y, w, h, scale):
    """A soft drop shadow (theme ``effects.shape_shadow``): the shape's blurred
    black silhouette, offset straight down, at the same blur/distance/alpha the
    .pptx builder writes as ``<a:outerShdw>`` (theme.SHAPE_SHADOW)."""
    spec = theme_mod.SHAPE_SHADOW
    blur, dist = spec["blur_pt"] * scale, spec["dist_pt"] * scale
    pad = int(math.ceil(blur * 3)) + 1
    gw, gh = max(1, int(round(w))), max(1, int(round(h)))
    mask = Image.new("L", (gw + 2 * pad, gh + 2 * pad), 0)
    _fill_shape_mask(ImageDraw.Draw(mask), name, el, pad, pad, gw, gh, scale)
    mask = mask.filter(ImageFilter.GaussianBlur(blur))
    alpha = spec["alpha"]
    mask = mask.point(lambda v: int(v * alpha))
    black = Image.new("RGB", mask.size, (0, 0, 0))
    img.paste(black, (int(round(x)) - pad, int(round(y + dist)) - pad), mask)


def _draw_shape(draw, theme, el, scale, img=None):
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

    # Deck-wide drop shadow (theme effects) goes under the shape, first.
    if img is not None and (theme.get("effects") or {}).get("shape_shadow"):
        _draw_shape_shadow(img, name, el, x, y, w, h, scale)

    # Linear-gradient fill: paint a masked gradient bitmap, then let the shape
    # dispatch below stroke only its outline (fill=None). Needs the target image.
    grad = el.get("gradient") or box.get("gradient")
    if grad and img is not None:
        c1, c2 = _gradient_colors(theme, grad)
        gw, gh = max(1, int(round(w))), max(1, int(round(h)))
        gimg = _gradient_rgb(gw, gh, c1, c2, grad.get("angle", 90.0))
        mask = Image.new("L", (gw, gh), 0)
        _fill_shape_mask(ImageDraw.Draw(mask), name, el, 0, 0, gw, gh, scale)
        img.paste(gimg, (int(round(x)), int(round(y))), mask)
        fill = None  # gradient already painted; shape dispatch strokes outline only

    if name in ("oval", "ellipse", "circle"):
        draw.ellipse([x, y, x2, y2], fill=fill, outline=o, width=ow)
    elif name == "rounded_rect":
        r = round_rect_radius_pt(el["w"], el["h"]) * scale
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
        # Lay the text out in the preset's inner text rect (an arrow's shaft, a
        # diamond's inscribed box…) exactly as PowerPoint does, then apply the
        # same frame insets the .pptx builder writes.
        _draw_text_frame(draw, theme, frame, shape_text_rect(name, x, y, w, h), scale,
                         insets=panel_text_insets(name, el["w"], el["h"]))


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


def render_harvey_png(color, value, size_px: int = 48) -> bytes:
    """Render a Harvey ball to transparent PNG bytes so the .pptx embeds it as a
    picture that pixel-matches the preview (OOXML pie-angle fills don't match)."""
    ss = 3
    n = max(16, int(size_px)) * ss
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    pad = n * 0.06
    _draw_harvey(ImageDraw.Draw(img), pad, pad, n - pad, n - pad, value, color, ss)
    img = img.resize((n // ss, n // ss), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


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


# Preset geometry below follows the OOXML presetShapeDefinitions at their DEFAULT
# adjustments — what python-pptx emits for MSO_SHAPE.* — so the preview matches
# the downloaded .pptx: block-arrow heads and the chevron/pentagon tips are half
# the SHORT side, hexagon corners a quarter of it, the star fills its box.
def _block_arrow_pts(x, y, w, h, name):
    head = min(w, h) * 0.5  # rightArrow/leftArrow adj2 = 50000 → 0.5 × ss
    shaft = h * 0.5         # adj1 = 50000 → shaft is half the height
    sy0, sy1 = y + (h - shaft) / 2, y + (h + shaft) / 2
    if name == "right_arrow":
        return [(x, sy0), (x + w - head, sy0), (x + w - head, y), (x + w, y + h / 2),
                (x + w - head, y + h), (x + w - head, sy1), (x, sy1)]
    return [(x + w, sy0), (x + head, sy0), (x + head, y), (x, y + h / 2),
            (x + head, y + h), (x + head, sy1), (x + w, sy1)]


def _up_down_arrow_pts(x, y, w, h, name):
    # upArrow/downArrow: the transposed twin of _block_arrow_pts.
    head = min(w, h) * 0.5
    shaft = w * 0.5
    sx0, sx1 = x + (w - shaft) / 2, x + (w + shaft) / 2
    if name == "up_arrow":
        return [(sx0, y + h), (sx0, y + head), (x, y + head), (x + w / 2, y),
                (x + w, y + head), (sx1, y + head), (sx1, y + h)]
    return [(sx0, y), (sx0, y + h - head), (x, y + h - head), (x + w / 2, y + h),
            (x + w, y + h - head), (sx1, y + h - head), (sx1, y)]


def _plus_pts(x, y, w, h):
    # OOXML mathPlus at its default adjustment (23520): the arms span 73.49% of
    # the box and are 23.52% of the short side thick, centred — a thin "+".
    cx, cy = x + w / 2, y + h / 2
    dx, dy = w * 0.36745, h * 0.36745
    t = min(w, h) * 0.1176
    return [(cx - t, cy - dy), (cx + t, cy - dy), (cx + t, cy - t), (cx + dx, cy - t),
            (cx + dx, cy + t), (cx + t, cy + t), (cx + t, cy + dy), (cx - t, cy + dy),
            (cx - t, cy + t), (cx - dx, cy + t), (cx - dx, cy - t), (cx - t, cy - t)]


def _cross_pts(x, y, w, h):
    # OOXML plus / MSO CROSS at its default adjustment (25000): a full-box Greek
    # cross whose corner notches are a quarter of the short side.
    i = min(w, h) * 0.25
    x1, x2, y1, y2 = x + i, x + w - i, y + i, y + h - i
    return [(x1, y), (x2, y), (x2, y1), (x + w, y1), (x + w, y2), (x2, y2),
            (x2, y + h), (x1, y + h), (x1, y2), (x, y2), (x, y1), (x1, y1)]


def _chevron_pts(x, y, w, h):
    # Match the OOXML preset geometry (default adj 50000): the notch/tip depth is
    # half the SHORT side, so a chain of chevrons interlocks in the preview
    # exactly as it does in the downloaded .pptx.
    notch = min(w, h) * 0.5
    return [(x, y), (x + w - notch, y), (x + w, y + h / 2), (x + w - notch, y + h),
            (x, y + h), (x + notch, y + h / 2)]


_STAR_HF, _STAR_VF, _STAR_INNER = 1.05146, 1.10557, 0.38196  # star5 hf / vf / adj÷50000


def _star_pts(x, y, w, h):
    # OOXML star5: the outer radii are stretched (hf/vf) and the centre pushed
    # down (svc = vc·vf) so the five points touch all four box edges; the inner
    # radius is the golden 0.382 of the outer.
    swd2, shd2 = (w / 2) * _STAR_HF, (h / 2) * _STAR_VF
    cx, cy = x + w / 2, y + (h / 2) * _STAR_VF
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        f = 1.0 if i % 2 == 0 else _STAR_INNER
        pts.append((cx + swd2 * f * math.cos(ang), cy + shd2 * f * math.sin(ang)))
    return pts


def _hexagon_pts(x, y, w, h):
    inset = min(w, h) * 0.25  # hexagon adj = 25000 → 0.25 × ss
    return [(x, y + h / 2), (x + inset, y), (x + w - inset, y),
            (x + w, y + h / 2), (x + w - inset, y + h), (x + inset, y + h)]


def _pentagon_pts(x, y, w, h):  # MSO "home plate" pentagon, pointing right
    tip = min(w, h) * 0.5  # homePlate adj = 50000 → 0.5 × ss
    return [(x, y), (x + w - tip, y), (x + w, y + h / 2), (x + w - tip, y + h), (x, y + h)]


def _diamond_pts(x, y, w, h):
    return [(x + w / 2, y), (x + w, y + h / 2), (x + w / 2, y + h), (x, y + h / 2)]


# Every polygon shape the preview draws. Together with rect/rounded_rect/oval
# (+ aliases) and harvey this MUST cover chat.slides.schema.SHAPE_NAMES — the
# schema rejects anything else so the preview never shows a shape as a plain
# rectangle that the .pptx would draw differently.
_POLY_SHAPES = {
    "diamond": _diamond_pts, "star": _star_pts, "hexagon": _hexagon_pts, "pentagon": _pentagon_pts,
    "right_arrow": lambda x, y, w, h: _block_arrow_pts(x, y, w, h, "right_arrow"),
    "left_arrow": lambda x, y, w, h: _block_arrow_pts(x, y, w, h, "left_arrow"),
    "up_arrow": lambda x, y, w, h: _up_down_arrow_pts(x, y, w, h, "up_arrow"),
    "down_arrow": lambda x, y, w, h: _up_down_arrow_pts(x, y, w, h, "down_arrow"),
    "chevron": _chevron_pts,
    "plus": _plus_pts, "cross": _cross_pts,
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


def _curve_points(x1, y1, x2, y2, curve, n=32):
    """Sample a quadratic Bézier whose control point is offset perpendicular to
    the chord midpoint by ``curve`` × chord length (the bow height / direction)."""
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    px, py = -dy / length, dx / length  # perpendicular unit vector
    cx, cy = mx + px * curve * length, my + py * curve * length
    pts = []
    for i in range(n + 1):
        t = i / n
        a, b, c = (1 - t) ** 2, 2 * (1 - t) * t, t * t
        pts.append((a * x1 + b * cx + c * x2, a * y1 + b * cy + c * y2))
    return pts


def _draw_line(draw, theme, el, scale):
    x1, y1 = el["x1"] * scale, el["y1"] * scale
    x2, y2 = el["x2"] * scale, el["y2"] * scale
    color = _rgb(theme, el.get("color"), (60, 60, 60))
    width = max(1, int((el.get("w") or 1) * scale))
    arrow = el.get("arrow", "none")
    curve = el.get("curve")
    if curve:
        pts = _curve_points(x1, y1, x2, y2, float(curve))
        draw.line(pts, fill=color, width=width, joint="curve")
        if arrow in ("end", "both"):
            _arrowhead(draw, pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], color, width)
        if arrow in ("start", "both"):
            _arrowhead(draw, pts[1][0], pts[1][1], pts[0][0], pts[0][1], color, width)
        return
    _styled_line(draw, (x1, y1), (x2, y2), color, width, el.get("dash"))
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


def _curve_bbox_pt(el):
    """Bounding box (ox, oy, w, h) in POINTS of a curved line, padded for the
    line width and arrowhead so the rasterised PNG isn't clipped."""
    pts = _curve_points(el["x1"], el["y1"], el["x2"], el["y2"], float(el.get("curve") or 0))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad = 10 + (el.get("w") or 1) * 2
    minx, miny = min(xs) - pad, min(ys) - pad
    return minx, miny, (max(xs) + pad) - minx, (max(ys) + pad) - miny


def render_line_png(theme, el, ss: int = 3):
    """Rasterise a curved line (+arrowheads) to a transparent PNG for the .pptx —
    python-pptx has no controllable-bow connector, so (like harvey/combo/icons)
    we embed a picture that pixel-matches the preview. Returns (png, bbox_pt)."""
    ox, oy, w, h = _curve_bbox_pt(el)
    img = Image.new("RGBA", (max(1, int(w * ss)), max(1, int(h * ss))), (0, 0, 0, 0))
    local = dict(el)
    local["x1"], local["y1"] = el["x1"] - ox, el["y1"] - oy
    local["x2"], local["y2"] = el["x2"] - ox, el["y2"] - oy
    _draw_line(ImageDraw.Draw(img), theme, local, ss)
    buf = BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue(), (ox, oy, w, h)


# ---------------------------------------------------------------------------
# Network / node-link diagrams (ecosystem maps, value webs, relationship graphs)
# ---------------------------------------------------------------------------
def _node_clearance(theme, el, nd) -> tuple[float, float]:
    """``(rx, ry)`` in points: how far an edge must stop from this node's centre.

    A dot node clears its radius; a centre-labelled node (``label_pos: "c"`` —
    the text-hub device, usually with ``r: 0``) clears the label's measured
    extents so edges stop at the words instead of striking through them."""
    emph = bool(nd.get("emphasis"))
    base_r = nd["r"] if nd.get("r") is not None else el.get("node_r", 5.0)
    r = base_r * (1.6 if emph else 1.0)
    label = nd.get("label") or ""
    if label and nd.get("label_pos", "r") == "c":
        fs = nd["size"] if nd.get("size") is not None else el.get("label_size", 12.0) * (1.25 if emph else 1.0)
        font = _font(theme_mod.font_family(theme, "body"), fs, emph, False)
        half_w = font.getlength(label) / 2 + 6
        half_h = fs * 0.75 + 4
        return (max(r + 2, half_w), max(r + 2, half_h))
    return (max(1.0, r + 2), max(1.0, r + 2))


def _clearance_t(theme, el, nd, dx, dy) -> float:
    """Fraction of the edge chord to trim at a node: where the chord leaves the
    node's clearance ellipse."""
    rx, ry = _node_clearance(theme, el, nd)
    q = (dx / rx) ** 2 + (dy / ry) ** 2
    return 0.0 if q <= 0 else min(0.5, 1.0 / math.sqrt(q))


def network_edge_segments(theme, el):
    """Valid, endpoint-trimmed edge segments for a network element, in POINTS
    (absolute slide coords): ``[(edge_dict, (x1, y1), (x2, y2))]``. Shared by the
    preview and the .pptx builder so both stop edges short of hub labels."""
    ox, oy = el.get("x", 0), el.get("y", 0)
    nodes = el.get("nodes") or []
    n = len(nodes)
    out = []
    for edge in el.get("edges") or []:
        a, b = edge.get("a"), edge.get("b")
        if not (isinstance(a, int) and isinstance(b, int) and 0 <= a < n and 0 <= b < n):
            continue
        ax, ay = ox + nodes[a]["x"], oy + nodes[a]["y"]
        bx, by = ox + nodes[b]["x"], oy + nodes[b]["y"]
        dx, dy = bx - ax, by - ay
        if not math.hypot(dx, dy):
            continue
        ta = _clearance_t(theme, el, nodes[a], dx, dy)
        tb = _clearance_t(theme, el, nodes[b], dx, dy)
        if ta + tb >= 1.0:  # clearances overlap (nodes too close) — keep the chord
            ta = tb = 0.0
        out.append((edge, (ax + dx * ta, ay + dy * ta), (bx - dx * tb, by - dy * tb)))
    return out


def _draw_network(draw, theme, el, scale, bg=None):
    """Draw a node-link diagram: edges under nodes, labels on top. Node coords
    are points relative to the element's ``x``/``y`` origin."""
    ox, oy = el["x"] * scale, el["y"] * scale
    nodes = el.get("nodes") or []

    e_color, e_w = el.get("edge_color", "accent2"), el.get("edge_w", 1.0)
    for edge, (x1, y1), (x2, y2) in network_edge_segments(theme, el):
        col = _rgb(theme, edge.get("color") or e_color, (120, 120, 120))
        w = max(1, int((edge.get("w") or e_w) * scale))
        _styled_line(draw, (x1 * scale, y1 * scale), (x2 * scale, y2 * scale), col, w, edge.get("dash"))

    n_color = el.get("node_color", "accent2")
    n_r = el.get("node_r", 5.0)
    # Default labels to light on a dark slide so they don't vanish; an explicit
    # label_color always wins.
    lbl_color = el.get("label_color") or ("lt1" if (bg is not None and _is_dark(bg)) else "dk1")
    lbl_size = el.get("label_size", 12.0)
    for nd in nodes:
        cx, cy = ox + nd["x"] * scale, oy + nd["y"] * scale
        emph = bool(nd.get("emphasis"))
        base_r = nd["r"] if nd.get("r") is not None else n_r
        r = base_r * (1.6 if emph else 1.0) * scale
        col = _rgb(theme, nd.get("color") or n_color, (80, 120, 100))
        if r > 0:
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        label = nd.get("label") or ""
        lpos = nd.get("label_pos", "r")
        if not label or lpos == "none":
            continue
        fs = nd["size"] if nd.get("size") is not None else lbl_size * (1.25 if emph else 1.0)
        lh = fs * 1.4 * scale
        lw = 180 * scale
        gap = (r if r > 0 else 0) + 4 * scale
        if lpos == "l":
            lx, ly, al = cx - gap - lw, cy - lh / 2, "right"
        elif lpos == "t":
            lx, ly, al = cx - lw / 2, cy - gap - lh, "center"
        elif lpos == "b":
            lx, ly, al = cx - lw / 2, cy + gap, "center"
        elif lpos == "c":
            lx, ly, al = cx - lw / 2, cy - lh / 2, "center"
        else:  # "r"
            lx, ly, al = cx + gap, cy - lh / 2, "left"
        frame = {"class": None, "valign": "middle", "paragraphs": [
            {"align": al, "runs": [{"t": label, "size": fs, "b": emph, "color": lbl_color}]}]}
        _draw_text_frame(draw, theme, frame, (lx, ly, lw, lh), scale)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _draw_table(draw, theme, el, scale, bg=None):
    rows = el.get("rows") or []
    n_cols = max((len(r) for r in rows), default=0)
    if not rows or n_cols == 0:  # no rows, or every row empty -> nothing to draw
        return
    dark_bg = bg is not None and _is_dark(bg)
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
            # text — a transparent (unfilled) cell sits on the slide bg, so on a
            # dark slide its default text must go light or it vanishes; a filled
            # (header/banded/explicit) cell keeps normal contrast.
            base = theme_mod.base_text_style(theme, spec.get("class") or text_class)
            if spec.get("color"):
                color = spec["color"]
            elif is_header:
                color = tbl.get("header_color")
            elif not fill_v and dark_bg:
                color = "lt1"
            else:
                color = base.get("color")
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
    op = _opacity(opacity)
    if op is None or op >= 1:
        return im
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


def _is_dark(rgb) -> bool:
    """Perceptual-luminance test so chart text/gridlines can flip for a dark slide."""
    r, g, b = rgb[0], rgb[1], rgb[2]
    return (0.299 * r + 0.587 * g + 0.114 * b) < 130


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
    # Axis/label/legend text is auto-drawn, so it must adapt to the slide bg —
    # on a dark or gradient slide, dark text would be invisible.
    dark_bg = bg is not None and _is_dark(bg)
    txt = _rgb(theme, "lt1" if dark_bg else "dk1", (235, 235, 235) if dark_bg else (30, 30, 30))
    axis = _rgb(theme, "accent3", (180, 180, 180))
    grid = tuple(round(a + (t - a) * 0.3) for a, t in zip(axis, txt)) if dark_bg \
        else tuple(min(255, c + 40) for c in axis)

    def fnt(px):
        return _font(theme_mod.font_family(theme, "data"), px * scale, False, False)

    pad = 8 * scale
    top = y + pad
    if el.get("title"):
        f = fnt(12)
        tw = f.getlength(el["title"])
        draw.text((x + (w - tw) / 2, top), el["title"], font=f, fill=txt)
        top += 22 * scale
    # histogram/bullet read only series[0] (extra series are ignored), so a
    # multi-series legend would be misleading — never legend those.
    show_legend = (kind in ("pie", "doughnut", "waterfall")
                   or (len(series) > 1 and kind not in ("histogram", "bullet"))) and el.get("legend", True)
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
    elif kind == "scatter":
        _chart_scatter(draw, plot, series, ramp, fnt, txt, grid, scale)
    elif kind == "histogram":
        _chart_histogram(draw, plot, series[0], el.get("bins"), ramp, fnt, txt, grid, el.get("value_labels"))
    elif kind == "dot":
        _chart_dot(draw, plot, series, cats, ramp, fnt, txt, grid, el.get("value_labels"), scale)
    elif kind == "bullet":
        _chart_bullet(draw, plot, series[0], cats, el.get("targets") or [], el.get("bands"), ramp, fnt, txt, el.get("value_labels"), scale)
    elif kind == "bar":
        _chart_bars(draw, plot, series, cats, ramp, fnt, txt, axis, grid, True, el.get("value_labels"), stacked, pt_colors)
    elif kind in ("line", "area"):
        _chart_lines(draw, plot, series, cats, ramp, fnt, txt, axis, grid, kind == "area", scale, bg)
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
            # Bars grow from the ZERO line (clamped into the plot), not from the
            # axis minimum — so negative values point the right way and every bar
            # is scaled correctly when the data spans zero.
            zfrac = min(1.0, max(0.0, (0.0 - lo) / span))
            vlbl = fnt(8)
            for si, s in enumerate(series):
                vals = s.get("values") or []
                v = vals[ci] if ci < len(vals) else 0
                if not isinstance(v, (int, float)):
                    continue
                frac = (v - lo) / span
                # Single-series bar highlighting: colour each bar by category.
                if point_colors and n_ser == 1 and ci < len(point_colors):
                    color = point_colors[ci]
                else:
                    color = ramp[si % len(ramp)]
                thick = slot * 0.7 / n_ser
                if horizontal:
                    by = ay + row * slot + slot * 0.15 + si * thick
                    x0, x1 = ax + min(frac, zfrac) * aw, ax + max(frac, zfrac) * aw
                    draw.rectangle([x0, by, x1, by + thick], fill=color)
                    if value_labels:
                        draw.text((x1 + 3, by + thick / 2 - 6), _fmt_num(v), font=vlbl, fill=txt)
                else:
                    bx = ax + row * slot + slot * 0.15 + si * thick
                    y0, y1 = ay + ah - max(frac, zfrac) * ah, ay + ah - min(frac, zfrac) * ah
                    draw.rectangle([bx, y0, bx + thick, y1], fill=color)
                    if value_labels:
                        lt = _fmt_num(v)
                        draw.text((bx + thick / 2 - vlbl.getlength(lt) / 2, y0 - 12), lt, font=vlbl, fill=txt)
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


def render_chart_png(theme, el, k: float = 3.0, dark_bg: bool = False) -> bytes:
    """Render a single chart element to transparent PNG bytes (for embedding a
    chart type the .pptx can't build natively — e.g. combo — as a picture).
    ``dark_bg`` flips axis/label text to light for a dark slide."""
    w_px = max(1, int(el.get("w", 480) * k))
    h_px = max(1, int(el.get("h", 300) * k))
    img = Image.new("RGBA", (w_px, h_px), (0, 0, 0, 0))
    local = dict(el)
    local["x"], local["y"] = 0.0, 0.0
    _draw_chart(ImageDraw.Draw(img), theme, local, k, bg=((20, 36, 27) if dark_bg else None))
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
    if band < 2:  # too many stages for the height -> drop gaps, split evenly
        gap, band = 0.0, max(1.0, ph / n)
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


def _chart_lines(draw, plot, series, cats, ramp, fnt, txt, axis, grid, area, scale, bg=None):
    px, py, pw, ph = plot
    # The preview surface is RGB, so a fill's alpha is ignored — pre-blend the
    # area fill toward the slide bg so it reads as a translucent wash (matching
    # PowerPoint) instead of a harsh, fully-saturated band.
    base = bg if (bg and len(bg) >= 3) else (255, 255, 255)

    def _wash(color):
        a = 0.28
        return tuple(round(b * (1 - a) + c * a) for b, c in zip(base, color[:3]))
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
            draw.polygon(poly, fill=_wash(color))
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
            # keep the label within the chart box (avoid clipping at the edges)
            tx = max(px, min(px + pw - tw, lx - tw / 2))
            draw.text((tx, ly - 6), pct, font=f, fill=(255, 255, 255))
        start += sweep
    if hole and hole > 0:
        hr = (d / 2) * hole
        draw.ellipse([cx - hr, cy - hr, cx + hr, cy + hr], fill=hole_bg or (255, 255, 255))
        if center_label:
            # Size to fill the hole, but shrink so a long label still fits inside.
            size_pt = min((hr * 2) / scale * 0.34, (hr * 2) / scale * 1.5 / max(1, len(center_label)))
            cf = fnt(max(10.0, size_pt))
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


# --- Shared compute helpers (also used by the .pptx builder so preview and
#     download bin/scale identically) ----------------------------------------
def _histogram_bins(values, bins=None):
    """``(edges, counts)`` for a histogram of raw ``values``. ``edges`` has
    ``len(counts)+1`` entries. ``bins`` is the bucket count (2–50); when None it
    defaults to ~sqrt(n), clamped to 5..20."""
    nums = [float(v) for v in (values or []) if isinstance(v, (int, float))]
    if not nums:
        return [], []
    lo, hi = min(nums), max(nums)
    if hi <= lo:
        hi = lo + 1.0
    if isinstance(bins, int) and bins >= 2:
        n = min(50, bins)
    else:
        n = max(5, min(20, int(round(math.sqrt(len(nums)))) or 5))
    width = (hi - lo) / n
    edges = [lo + i * width for i in range(n + 1)]
    counts = [0] * n
    for v in nums:
        idx = int((v - lo) / width)
        counts[min(idx, n - 1)] += 1  # the max value lands in the last bin
    return edges, counts


def _scatter_bounds(series):
    """``(xlo, xhi, ylo, yhi)`` padded data ranges for a scatter/bubble chart."""
    xs, ys = [], []
    for s in series:
        for p in (s.get("points") or []):
            if isinstance(p, list) and len(p) >= 2 and all(isinstance(c, (int, float)) for c in p[:2]):
                xs.append(float(p[0]))
                ys.append(float(p[1]))
    if not xs:
        return (0.0, 1.0, 0.0, 1.0)

    def _pad(lo, hi):
        if hi <= lo:
            hi = lo + 1.0
        m = (hi - lo) * 0.08
        return lo - m, hi + m

    xlo, xhi = _pad(min(xs), max(xs))
    ylo, yhi = _pad(min(ys), max(ys))
    return (xlo, xhi, ylo, yhi)


def _chart_scatter(draw, plot, series, ramp, fnt, txt, grid, scale):
    """An x/y scatter (relationship between two measures). A point with a 3rd
    value is drawn as a bubble whose radius scales with that value."""
    px, py, pw, ph = plot
    xlo, xhi, ylo, yhi = _scatter_bounds(series)
    xspan, yspan = (xhi - xlo) or 1.0, (yhi - ylo) or 1.0
    lbl = fnt(9)
    val_gutter, cat_gutter = 30, 18
    ax, ay, aw, ah = px + val_gutter, py, pw - val_gutter, ph - cat_gutter

    def gx_(v):
        return ax + (v - xlo) / xspan * aw

    def gy_(v):
        return ay + ah - (v - ylo) / yspan * ah

    _, ystep = _nice_ticks(yhi if yhi > 0 else 1.0)
    t = math.ceil(ylo / ystep) * ystep
    while t <= yhi + 1e-9:
        gy = gy_(t)
        draw.line([(ax, gy), (ax + aw, gy)], fill=grid, width=1)
        draw.text((px, gy - 6), _fmt_num(t), font=lbl, fill=txt)
        t += ystep
    _, xstep = _nice_ticks(xhi if xhi > 0 else 1.0)
    t = math.ceil(xlo / xstep) * xstep
    while t <= xhi + 1e-9:
        gx = gx_(t)
        draw.text((gx - lbl.getlength(_fmt_num(t)) / 2, ay + ah + 3), _fmt_num(t), font=lbl, fill=txt)
        t += xstep

    sizes = [float(p[2]) for s in series for p in (s.get("points") or [])
             if isinstance(p, list) and len(p) >= 3 and isinstance(p[2], (int, float))]
    smax = max(sizes) if sizes else 0.0
    base_r = max(2.5, 3.0 * scale)
    for si, s in enumerate(series):
        color = ramp[si % len(ramp)]
        for p in (s.get("points") or []):
            if not (isinstance(p, list) and len(p) >= 2):
                continue
            cx, cy = gx_(float(p[0])), gy_(float(p[1]))
            if len(p) >= 3 and smax > 0 and isinstance(p[2], (int, float)):
                r = base_r + (max(0.0, float(p[2])) / smax) * (18 * scale)  # bubble
            else:
                r = base_r
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)


def _chart_histogram(draw, plot, series, bins, ramp, fnt, txt, grid, value_labels):
    """A histogram: raw ``series['values']`` bucketed into equal-width, contiguous
    (gapless) columns of counts."""
    px, py, pw, ph = plot
    edges, counts = _histogram_bins(series.get("values"), bins)
    if not counts:
        return
    top, step = _nice_ticks(max(counts) or 1)
    span = top or 1.0
    lbl = fnt(9)
    val_gutter, cat_gutter = 26, 16
    ax, ay, aw, ah = px + val_gutter, py, pw - val_gutter, ph - cat_gutter
    t = 0
    while t <= top + 1e-9:
        gy = ay + ah - (t / span) * ah
        draw.line([(ax, gy), (ax + aw, gy)], fill=grid, width=1)
        draw.text((px, gy - 6), _fmt_num(t), font=lbl, fill=txt)
        t += step
    n = len(counts)
    color = ramp[0]
    slot = aw / n
    for i, c in enumerate(counts):
        bh = (c / span) * ah
        x0 = ax + i * slot
        draw.rectangle([x0, ay + ah - bh, x0 + slot, ay + ah], fill=color,
                       outline=(255, 255, 255), width=1)  # hairline separates bins
        if value_labels and c > 0:
            cw = lbl.getlength(str(c))
            draw.text((x0 + slot / 2 - cw / 2, ay + ah - bh - 12), str(c), font=lbl, fill=txt)
    everyk = max(1, round((n + 1) / 6))
    for i in range(0, n + 1, everyk):
        lt = _fmt_num(edges[i])
        draw.text((ax + i * slot - lbl.getlength(lt) / 2, ay + ah + 3), lt, font=lbl, fill=txt)


def _chart_dot(draw, plot, series, cats, ramp, fnt, txt, grid, value_labels, scale):
    """A horizontal dot plot / lollipop — a stem + a dot at each category's value.
    Sort the categories upstream for a ranked dot plot."""
    px, py, pw, ph = plot
    n_cat = len(cats)
    if n_cat == 0 or not series:
        return
    lo, hi = _val_axis(series)
    span = (hi - lo) or 1.0
    lbl = fnt(9)
    cat_gutter = 30
    ax, ay, aw, ah = px + cat_gutter, py, pw - cat_gutter, ph - 16
    _, step = _nice_ticks(hi if hi > 0 else 1.0)
    t = lo
    while t <= hi + 1e-9:
        gx = ax + (t - lo) / span * aw
        draw.line([(gx, ay), (gx, ay + ah)], fill=grid, width=1)
        draw.text((gx - lbl.getlength(_fmt_num(t)) / 2, ay + ah + 3), _fmt_num(t), font=lbl, fill=txt)
        t += step
    zx = ax + (0.0 - lo) / span * aw  # baseline x for the stems
    dot_r = max(3.0, 4.0 * scale)
    for ci in range(n_cat):
        cy = ay + (ci + 0.5) * (ah / n_cat)
        clbl = cats[ci]
        draw.text((px + cat_gutter - lbl.getlength(clbl) - 4, cy - 6), clbl, font=lbl, fill=txt)
        for si, s in enumerate(series):
            vals = s.get("values") or []
            v = vals[ci] if ci < len(vals) else 0
            if not isinstance(v, (int, float)):
                continue
            gx = ax + (v - lo) / span * aw
            color = ramp[si % len(ramp)]
            draw.line([(zx, cy), (gx, cy)], fill=color, width=max(1, int(scale)))
            draw.ellipse([gx - dot_r, cy - dot_r, gx + dot_r, cy + dot_r], fill=color)
            if value_labels:
                draw.text((gx + dot_r + 3, cy - 6), _fmt_num(v), font=lbl, fill=txt)


def _bullet_bands(bands):
    """Qualitative-band thresholds as ascending FRACTIONS of each row's target
    (default 50/75/100%). A list that looks absolute (any value > 2) is treated as
    unset, so a mis-authored bullet still renders sensible bands."""
    fracs = [float(b) for b in (bands or []) if isinstance(b, (int, float))]
    if not fracs or max(fracs) > 2.0:
        fracs = [0.5, 0.75, 1.0]
    return sorted(min(2.0, max(0.0, f)) for f in fracs)


def _chart_bullet(draw, plot, series, cats, targets, bands, ramp, fnt, txt, value_labels, scale):
    """A bullet chart. Each row is scaled to its OWN target (KPIs carry different
    units), so the qualitative ``bands`` — FRACTIONS of that row's target — stretch
    across the row; a measure bar and a target tick sit on top. The category label
    sits ABOVE its bar, so a long name can never overflow the slide edge."""
    px, py, pw, ph = plot
    vals = series.get("values") or []
    n = len(cats) if cats else len(vals)
    if n == 0:
        return
    lbl = fnt(9)
    fracs = _bullet_bands(bands)
    top_frac = fracs[-1] if fracs else 1.0
    ax, ay, aw, ah = px, py, pw, ph
    row_h = ah / n
    label_h = min(16 * scale, row_h * 0.42)
    for ci in range(n):
        v = vals[ci] if ci < len(vals) else 0
        v = float(v) if isinstance(v, (int, float)) else 0.0
        tgt = float(targets[ci]) if (ci < len(targets) and isinstance(targets[ci], (int, float))) else (v or 1.0)
        row_max = (max(v, tgt, top_frac * tgt) * 1.12) or 1.0
        y0 = ay + ci * row_h
        # label ABOVE the bar (full width; cannot overflow the left edge)
        draw.text((ax, y0 + 1), (cats[ci] if ci < len(cats) else ""), font=lbl, fill=txt)
        band_top = y0 + label_h
        band_h = min(row_h - label_h - 6, 24 * scale)
        cy = band_top + band_h / 2
        # qualitative zones: light -> dark, as fractions of the row's target
        prev = 0.0
        for bi, f in enumerate(fracs):
            x0 = ax + (prev * tgt) / row_max * aw
            x1 = ax + (f * tgt) / row_max * aw
            g = 232 - int(bi / max(1, len(fracs)) * 70)
            draw.rectangle([x0, band_top, x1, band_top + band_h], fill=(g, g, g))
            prev = f
        # measure bar (thinner, drawn over the bands)
        mh = band_h * 0.46
        draw.rectangle([ax, cy - mh / 2, ax + v / row_max * aw, cy + mh / 2], fill=ramp[0])
        # target tick
        tx = ax + tgt / row_max * aw
        draw.line([(tx, band_top - 2), (tx, band_top + band_h + 2)], fill=txt, width=max(2, int(2 * scale)))
        if value_labels:
            draw.text((min(ax + v / row_max * aw + 3, ax + aw - 26 * scale), cy - 6), _fmt_num(v), font=lbl, fill=txt)


# ---------------------------------------------------------------------------
# Footer / page number
# ---------------------------------------------------------------------------
def _footer_text_pillow(draw, font, txt, x, w, y0, band_h, align, fill, scale):
    tw = font.getlength(txt)
    asc, desc = font.getmetrics()
    ty = (y0 + band_h / 2) * scale - (asc + desc) / 2
    if align == "center":
        tx = (x + w / 2) * scale - tw / 2
    elif align == "right":
        tx = (x + w) * scale - tw
    else:
        tx = x * scale
    draw.text((tx, ty), txt, font=font, fill=fill)


def _footer_logo_pillow(img, logo_bytes, x, w, y0, band_h, align, scale, logo_height=None):
    if not logo_bytes:
        return
    try:
        src = Image.open(BytesIO(logo_bytes)).convert("RGBA")
    except Exception:  # noqa: BLE001
        logger.warning("pillow footer logo failed", exc_info=True)
        return
    sw, sh = src.size
    if not sw or not sh:
        return
    lh_pt, top_pt = theme_mod.footer_logo_box(logo_height, y0, band_h)
    lh = max(1, int(round(lh_pt * scale)))
    lw = max(1, int(sw * (lh / sh)))
    placed = src.resize((lw, lh))
    if align == "center":
        px = int((x + w / 2) * scale - lw / 2)
    elif align == "right":
        px = int((x + w) * scale - lw)
    else:
        px = int(x * scale)
    img.paste(placed, (px, int(round(top_pt * scale))), placed)


def _stamp_footer(draw, img, theme, scale, page_num, total, bg=None, footer_logo=None):
    """Footer band: an optional full-bleed background plus three grid sections
    (logo / disclaimer text / page number), each at its colspan + alignment."""
    from chat.slides.layouts import footer_section_box

    footer = theme.get("footer", {}) or {}
    band_h = theme_mod.FOOTER_BAND_H
    y0 = 540 - band_h

    bgc = footer.get("bg_color") or ""
    if bgc:
        band_rgb = _rgb(theme, bgc, (236, 239, 233))
        band_dark = _is_dark(band_rgb)
        draw.rectangle([0, y0 * scale, img.width, 540 * scale], fill=band_rgb)
    else:
        band_dark = bg is not None and _is_dark(bg)

    size = footer.get("size", 9)
    text_color = "lt2" if band_dark else footer.get("color", "dk2")
    fill = _rgb(theme, text_color)
    font = _font(theme_mod.font_family(theme, "data"), size * scale, False, False)

    start_col = 1
    for sec in footer.get("sections", []) or []:
        colspan = int(sec.get("colspan", 4) or 4)
        content = sec.get("content", "none")
        align = sec.get("align", "left")
        if content and content != "none" and start_col <= 12:
            x, w = footer_section_box(start_col, colspan)
            if content == "logo":
                _footer_logo_pillow(img, footer_logo, x, w, y0, band_h, align, scale, footer.get("logo_height"))
            else:
                txt = footer.get("text", "") if content == "text" else str(page_num)
                if txt:
                    _footer_text_pillow(draw, font, txt, x, w, y0, band_h, align, fill, scale)
        start_col += colspan


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
        op = _opacity(el.get("opacity"))
        if op is not None and op < 1:  # translucent panel: draw on a layer, fade, composite
            layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            _draw_shape(ImageDraw.Draw(layer), theme, el, scale, img=layer)
            layer = _apply_opacity(layer, op)
            img.paste(layer, (0, 0), layer)
        else:
            _draw_shape(draw, theme, el, scale, img=img)
    elif etype == "icon":
        op = _opacity(el.get("opacity"))
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
        _draw_table(draw, theme, el, scale, bg)
    elif etype == "line":
        _draw_line(draw, theme, el, scale)
    elif etype == "chart":
        _draw_chart(draw, theme, el, scale, bg)
    elif etype == "network":
        _draw_network(draw, theme, el, scale, bg)


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
                     supersample: int = 2, font_book=None) -> tuple[bytes, int, int]:
    """Render slide ``index`` of ``deck`` to ``(png_bytes, width_px, height_px)``."""
    theme = theme_mod.resolve_theme(deck)
    if font_book is not None:
        theme["_fontbook"] = font_book   # picked up by _run_font (transient, per-render)
    size = deck.get("size") or {}
    w_pt, h_pt = size.get("w", 960), size.get("h", 540)
    scale = (dpi / 72.0) * supersample
    W, H = int(round(w_pt * scale)), int(round(h_pt * scale))
    slides = deck.get("slides") or []
    slide = slides[index]

    bg = _rgb(theme, slide.get("bg") or "lt1", (255, 255, 255))
    img = Image.new("RGB", (W, H), bg)

    # Full-bleed gradient background (behind everything, over the flat bg fill).
    bgrad = slide.get("bg_gradient")
    if bgrad:
        c1, c2 = _gradient_colors(theme, bgrad)
        img.paste(_gradient_rgb(W, H, c1, c2, bgrad.get("angle", 90.0)), (0, 0))
        # Text-contrast decisions use the gradient's mean, not one end, so a
        # gradient that reverses lightness doesn't mis-pick contrast.
        bg = tuple(round((a + b) / 2) for a, b in zip(c1, c2))

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
        # The scrim is the backdrop elements sit on, so fold it into `bg` — this
        # is what lets a chart/table/network over a dark photo pick light text.
        bg = tuple(round(b * (1 - op) + c * op) for b, c in zip(bg, col))

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
        footer_logo = None
        _tid, _ext = theme.get("_theme_id"), theme.get("_theme_logo_ext")
        if _tid and _ext:
            try:
                from chat.slides.logos import slide_theme_logo_bytes

                footer_logo = slide_theme_logo_bytes(_tid, _ext)
            except Exception:  # noqa: BLE001
                footer_logo = None
        _stamp_footer(draw, img, theme, scale, index + 1, len(slides), bg, footer_logo)

    if supersample > 1:
        img = img.resize((int(round(w_pt * dpi / 72.0)), int(round(h_pt * dpi / 72.0))), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), img.width, img.height


def render_deck_pngs(deck: dict, *, dpi: int = 120, image_resolver=None,
                     only_slide_ids=None, font_faces=None) -> list[tuple[str, bytes, int, int]]:
    """Render a deck to ``[(slide_id, png_bytes, w, h)]``. ``only_slide_ids`` limits it.

    ``font_faces`` is the org-resolved face map from
    :func:`chat.slides.font_resolve.resolve_deck_font_faces` (bundled + uploaded); when
    given, uploaded brand fonts rasterise from their bytes instead of falling back to a
    bundled face. ``None`` keeps the historical bundled-on-disk behaviour.
    """
    slides = deck.get("slides") or []
    only = set(only_slide_ids) if only_slide_ids else None
    book = _FontBook(font_faces) if font_faces else None
    out = []
    for i, s in enumerate(slides):
        sid = s.get("id") or f"_{i}"
        if only is not None and sid not in only:
            continue
        png, w, h = render_slide_png(deck, i, dpi=dpi, image_resolver=image_resolver, font_book=book)
        out.append((sid, png, w, h))
    return out
