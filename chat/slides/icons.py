"""A small, curated library of business/consulting icons drawn procedurally.

Each icon is a function that strokes/fills a normalised 0..1 box with PIL's
``ImageDraw`` in a single colour. One definition serves both render paths: the
Pillow preview draws it directly, and the .pptx build rasterises it to a
transparent, theme-tinted PNG (``render_icon_png``) and embeds it as a picture.

Why not an icon font or SVG? A font doesn't travel reliably inside a .pptx, and
SVG needs a native renderer we deliberately avoid. Hand-stroked primitives stay
dependency-free, tint to any theme colour, and look identical in both paths.

Coordinates are 0..1 within the icon's box; the caller pads the box and picks
the stroke width. Add an icon = add one function + one ``ICONS`` entry.
"""

from __future__ import annotations

import math
from io import BytesIO


def _pt(box, u, v):
    x0, y0, x1, y1 = box
    return (x0 + u * (x1 - x0), y0 + v * (y1 - y0))


def _line(draw, box, pts, color, sw):
    draw.line([_pt(box, u, v) for u, v in pts], fill=color, width=sw, joint="curve")


def _poly(draw, box, pts, color, sw, fill=None, closed=True):
    p = [_pt(box, u, v) for u, v in pts]
    if fill is not None:
        draw.polygon(p, fill=fill, outline=color)
    elif closed:
        draw.line(p + [p[0]], fill=color, width=sw, joint="curve")
    else:
        draw.line(p, fill=color, width=sw, joint="curve")


def _ellipse(draw, box, cx, cy, rx, ry, color, sw, fill=None):
    x0, y0 = _pt(box, cx - rx, cy - ry)
    x1, y1 = _pt(box, cx + rx, cy + ry)
    draw.ellipse([x0, y0, x1, y1], outline=(None if fill is not None else color),
                 width=sw, fill=fill)


def _arc(draw, box, cx, cy, rx, ry, a0, a1, color, sw):
    x0, y0 = _pt(box, cx - rx, cy - ry)
    x1, y1 = _pt(box, cx + rx, cy + ry)
    draw.arc([x0, y0, x1, y1], a0, a1, fill=color, width=sw)


def _dot(draw, box, cx, cy, r, color):
    _ellipse(draw, box, cx, cy, r, r, color, 1, fill=color)


def _arrowhead(draw, box, tip, ang, color, size=0.16):
    tx, ty = tip
    c, s = math.cos(ang), math.sin(ang)
    back = (tx - size * c, ty - size * s)
    wing = size * 0.62
    left = (back[0] + wing * s, back[1] - wing * c)
    right = (back[0] - wing * s, back[1] + wing * c)
    draw.polygon([_pt(box, *tip), _pt(box, *left), _pt(box, *right)], fill=color)


# --- icon definitions -------------------------------------------------------

def _check(d, b, c, sw):
    _line(d, b, [(0.16, 0.55), (0.42, 0.80), (0.85, 0.22)], c, sw)


def _x(d, b, c, sw):
    _line(d, b, [(0.24, 0.24), (0.76, 0.76)], c, sw)
    _line(d, b, [(0.76, 0.24), (0.24, 0.76)], c, sw)


def _plus(d, b, c, sw):
    _line(d, b, [(0.5, 0.18), (0.5, 0.82)], c, sw)
    _line(d, b, [(0.18, 0.5), (0.82, 0.5)], c, sw)


def _minus(d, b, c, sw):
    _line(d, b, [(0.18, 0.5), (0.82, 0.5)], c, sw)


def _arrow_right(d, b, c, sw):
    _line(d, b, [(0.16, 0.5), (0.78, 0.5)], c, sw)
    _arrowhead(d, b, (0.84, 0.5), 0.0, c)


def _arrow_left(d, b, c, sw):
    _line(d, b, [(0.84, 0.5), (0.22, 0.5)], c, sw)
    _arrowhead(d, b, (0.16, 0.5), math.pi, c)


def _arrow_up(d, b, c, sw):
    _line(d, b, [(0.5, 0.84), (0.5, 0.22)], c, sw)
    _arrowhead(d, b, (0.5, 0.16), -math.pi / 2, c)


def _arrow_down(d, b, c, sw):
    _line(d, b, [(0.5, 0.16), (0.5, 0.78)], c, sw)
    _arrowhead(d, b, (0.5, 0.84), math.pi / 2, c)


def _trend_up(d, b, c, sw):
    _line(d, b, [(0.14, 0.72), (0.40, 0.50), (0.56, 0.60), (0.84, 0.26)], c, sw)
    _arrowhead(d, b, (0.86, 0.24), math.atan2(0.26 - 0.60, 0.84 - 0.56), c, size=0.18)


def _trend_down(d, b, c, sw):
    _line(d, b, [(0.14, 0.28), (0.40, 0.50), (0.56, 0.40), (0.84, 0.74)], c, sw)
    _arrowhead(d, b, (0.86, 0.76), math.atan2(0.74 - 0.40, 0.84 - 0.56), c, size=0.18)


def _target(d, b, c, sw):
    _ellipse(d, b, 0.5, 0.5, 0.42, 0.42, c, sw)
    _ellipse(d, b, 0.5, 0.5, 0.24, 0.24, c, sw)
    _dot(d, b, 0.5, 0.5, 0.08, c)


def _check_circle(d, b, c, sw):
    _ellipse(d, b, 0.5, 0.5, 0.42, 0.42, c, sw)
    _line(d, b, [(0.30, 0.52), (0.45, 0.66), (0.72, 0.36)], c, sw)


def _warning(d, b, c, sw):
    _poly(d, b, [(0.5, 0.14), (0.9, 0.84), (0.1, 0.84)], c, sw)
    _line(d, b, [(0.5, 0.40), (0.5, 0.62)], c, sw)
    _dot(d, b, 0.5, 0.74, 0.045, c)


def _info(d, b, c, sw):
    _ellipse(d, b, 0.5, 0.5, 0.42, 0.42, c, sw)
    _dot(d, b, 0.5, 0.30, 0.05, c)
    _line(d, b, [(0.5, 0.44), (0.5, 0.72)], c, sw)


def _star(d, b, c, sw):
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        r = 0.44 if i % 2 == 0 else 0.18
        pts.append((0.5 + r * math.cos(ang), 0.5 + r * math.sin(ang)))
    _poly(d, b, pts, c, sw, fill=c)


def _clock(d, b, c, sw):
    _ellipse(d, b, 0.5, 0.5, 0.42, 0.42, c, sw)
    _line(d, b, [(0.5, 0.5), (0.5, 0.26)], c, sw)
    _line(d, b, [(0.5, 0.5), (0.68, 0.58)], c, sw)


def _calendar(d, b, c, sw):
    x0, y0 = _pt(b, 0.14, 0.20)
    x1, y1 = _pt(b, 0.86, 0.86)
    d.rounded_rectangle([x0, y0, x1, y1], radius=(x1 - x0) * 0.10, outline=c, width=sw)
    _line(d, b, [(0.14, 0.36), (0.86, 0.36)], c, sw)
    _line(d, b, [(0.32, 0.12), (0.32, 0.26)], c, sw)
    _line(d, b, [(0.68, 0.12), (0.68, 0.26)], c, sw)


def _bar_chart(d, b, c, sw):
    for u, top in ((0.20, 0.62), (0.44, 0.40), (0.68, 0.22)):
        x0, y0 = _pt(b, u, top)
        x1, y1 = _pt(b, u + 0.14, 0.84)
        d.rectangle([x0, y0, x1, y1], fill=c)


def _shield(d, b, c, sw):
    _poly(d, b, [(0.5, 0.12), (0.84, 0.26), (0.84, 0.54),
                 (0.5, 0.88), (0.16, 0.54), (0.16, 0.26)], c, sw)


def _location(d, b, c, sw):
    _arc(d, b, 0.5, 0.42, 0.30, 0.30, 150, 390, c, sw)
    _line(d, b, [(0.24, 0.53), (0.5, 0.88)], c, sw)
    _line(d, b, [(0.76, 0.53), (0.5, 0.88)], c, sw)
    _dot(d, b, 0.5, 0.42, 0.09, c)


def _lightbulb(d, b, c, sw):
    _ellipse(d, b, 0.5, 0.40, 0.30, 0.30, c, sw)
    _line(d, b, [(0.40, 0.66), (0.40, 0.78)], c, sw)
    _line(d, b, [(0.60, 0.66), (0.60, 0.78)], c, sw)
    _line(d, b, [(0.40, 0.80), (0.60, 0.80)], c, sw)
    _line(d, b, [(0.42, 0.88), (0.58, 0.88)], c, sw)


def _gear(d, b, c, sw):
    outer, inner = 0.44, 0.34
    pts = []
    teeth = 8
    for i in range(teeth * 2):
        ang = i * math.pi / teeth
        r = outer if i % 2 == 0 else inner
        pts.append((0.5 + r * math.cos(ang), 0.5 + r * math.sin(ang)))
    _poly(d, b, pts, c, sw)
    _ellipse(d, b, 0.5, 0.5, 0.14, 0.14, c, sw)


def _person(d, b, c, sw):
    _ellipse(d, b, 0.5, 0.30, 0.17, 0.17, c, sw)
    _arc(d, b, 0.5, 0.92, 0.32, 0.40, 180, 360, c, sw)


def _people(d, b, c, sw):
    _ellipse(d, b, 0.36, 0.32, 0.14, 0.14, c, sw)
    _arc(d, b, 0.36, 0.90, 0.26, 0.34, 180, 360, c, sw)
    _ellipse(d, b, 0.68, 0.34, 0.12, 0.12, c, sw)
    _arc(d, b, 0.70, 0.90, 0.22, 0.30, 200, 360, c, sw)


def _building(d, b, c, sw):
    x0, y0 = _pt(b, 0.20, 0.16)
    x1, y1 = _pt(b, 0.80, 0.86)
    d.rectangle([x0, y0, x1, y1], outline=c, width=sw)
    for row in (0.30, 0.48, 0.66):
        for col in (0.32, 0.50, 0.68):
            _dot(d, b, col, row, 0.035, c)


def _globe(d, b, c, sw):
    _ellipse(d, b, 0.5, 0.5, 0.42, 0.42, c, sw)
    _ellipse(d, b, 0.5, 0.5, 0.18, 0.42, c, sw)
    _line(d, b, [(0.08, 0.5), (0.92, 0.5)], c, sw)
    _arc(d, b, 0.5, 0.16, 0.42, 0.30, 20, 160, c, sw)
    _arc(d, b, 0.5, 0.84, 0.42, 0.30, 200, 340, c, sw)


def _flag(d, b, c, sw):
    _line(d, b, [(0.26, 0.12), (0.26, 0.88)], c, sw)
    _poly(d, b, [(0.26, 0.16), (0.80, 0.28), (0.26, 0.44)], c, sw, fill=c)


def _search(d, b, c, sw):
    _ellipse(d, b, 0.42, 0.42, 0.28, 0.28, c, sw)
    _line(d, b, [(0.63, 0.63), (0.86, 0.86)], c, sw)


def _bolt(d, b, c, sw):
    _poly(d, b, [(0.56, 0.10), (0.26, 0.56), (0.46, 0.56),
                 (0.40, 0.90), (0.74, 0.42), (0.52, 0.42)], c, sw, fill=c)


def _cash(d, b, c, sw):
    x0, y0 = _pt(b, 0.12, 0.28)
    x1, y1 = _pt(b, 0.88, 0.72)
    d.rounded_rectangle([x0, y0, x1, y1], radius=(y1 - y0) * 0.14, outline=c, width=sw)
    _ellipse(d, b, 0.5, 0.5, 0.11, 0.11, c, sw)


def _rocket(d, b, c, sw):
    _poly(d, b, [(0.5, 0.10), (0.66, 0.42), (0.66, 0.66),
                 (0.34, 0.66), (0.34, 0.42)], c, sw)
    _dot(d, b, 0.5, 0.36, 0.06, c)
    _line(d, b, [(0.34, 0.55), (0.20, 0.72)], c, sw)
    _line(d, b, [(0.20, 0.72), (0.34, 0.66)], c, sw)
    _line(d, b, [(0.66, 0.55), (0.80, 0.72)], c, sw)
    _line(d, b, [(0.80, 0.72), (0.66, 0.66)], c, sw)
    _line(d, b, [(0.44, 0.72), (0.5, 0.88), (0.56, 0.72)], c, sw)


def _mail(d, b, c, sw):
    x0, y0 = _pt(b, 0.12, 0.24)
    x1, y1 = _pt(b, 0.88, 0.76)
    d.rectangle([x0, y0, x1, y1], outline=c, width=sw)
    _line(d, b, [(0.12, 0.24), (0.5, 0.54), (0.88, 0.24)], c, sw)


def _doc(d, b, c, sw):
    _poly(d, b, [(0.26, 0.12), (0.62, 0.12), (0.74, 0.26),
                 (0.74, 0.88), (0.26, 0.88)], c, sw)
    for v in (0.44, 0.58, 0.72):
        _line(d, b, [(0.36, v), (0.64, v)], c, max(1, sw - 1))


ICONS = {
    "check": _check, "x": _x, "close": _x, "plus": _plus, "minus": _minus,
    "arrow_right": _arrow_right, "arrow_left": _arrow_left,
    "arrow_up": _arrow_up, "arrow_down": _arrow_down,
    "trend_up": _trend_up, "trend_down": _trend_down,
    "target": _target, "check_circle": _check_circle, "warning": _warning,
    "info": _info, "star": _star, "clock": _clock, "calendar": _calendar,
    "bar_chart": _bar_chart, "shield": _shield, "location": _location,
    "lightbulb": _lightbulb, "gear": _gear, "person": _person, "people": _people,
    "building": _building, "globe": _globe, "flag": _flag, "search": _search,
    "bolt": _bolt, "cash": _cash, "rocket": _rocket, "mail": _mail, "doc": _doc,
}

# Public names (deduped, stable order) for the skill catalogue and validation.
ICON_NAMES = sorted(set(ICONS))


def draw_icon(draw, name, box, color, sw) -> bool:
    """Stroke icon ``name`` into ``box`` (pixels) in ``color``. Returns False if
    the name is unknown (caller may draw a fallback)."""
    fn = ICONS.get(name)
    if fn is None:
        return False
    fn(draw, box, color, max(1, int(sw)))
    return True


def render_icon_png(name, size_px, color, sw=None) -> bytes | None:
    """Render an icon to transparent-background PNG bytes (for .pptx embedding)."""
    from PIL import Image, ImageDraw

    if name not in ICONS:
        return None
    ss = 3  # supersample for crisp edges, then downscale
    n = max(16, int(size_px)) * ss
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = n * 0.08
    stroke = int(round((sw if sw else max(2.0, size_px * 0.09)) * ss))
    draw_icon(d, name, (pad, pad, n - pad, n - pad), color, stroke)
    img = img.resize((n // ss, n // ss), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()
