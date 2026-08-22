"""Seed slide JSON for ``slides_add_slide(layout=...)``.

Each entry is a ready-to-edit single slide on the 960x540 grid (48pt side
margins -> 864pt content width). The tool deep-copies the seed, mints fresh
ids, and inserts it; the model then edits the placeholder text freely. Seeds use
theme text CLASSES and colour NAMES, never literals, so they inherit branding.
"""

from __future__ import annotations

import copy

# Shared grid constants (points).
_ML = 48          # left margin
_MW = 864         # content width (960 - 2*48)
_MR = 912         # right edge of content


def _text(x, y, w, h, cls, paragraphs, **extra):
    el = {"type": "text", "x": x, "y": y, "w": w, "h": h, "class": cls, "paragraphs": paragraphs}
    el.update(extra)
    return el


def _p(text, color=None, size=None, bold=None, **kw):
    run = {"t": text}
    if color:
        run["color"] = color
    if size:
        run["size"] = size
    if bold is not None:
        run["b"] = bold
    para = {"runs": [run]}
    para.update(kw)
    return para


_LAYOUTS: dict[str, dict] = {
    "title": {
        "name": "Title",
        "skip_footer": True,
        "bg": "lt1",
        "elements": [
            _text(_ML, 210, _MW, 90, "headline", [_p("Presentation title", align="center")], valign="bottom"),
            _text(_ML, 305, _MW, 40, "subhead", [_p("Subtitle or author · date", align="center")]),
        ],
    },
    "section": {
        "name": "Section",
        "bg": "dk2",
        "elements": [
            _text(_ML, 230, _MW, 80, "headline", [_p("Section heading", color="lt1", align="left")]),
            {"type": "line", "x1": _ML, "y1": 220, "x2": _ML + 120, "y2": 220, "color": "accent1", "w": 3},
        ],
    },
    "bullets": {
        "name": "Bullets",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Slide title")]),
            _text(_ML, 130, _MW, 340, "body", [
                _p("First key point", bullet=True, space_after=10),
                _p("Second key point", bullet=True, space_after=10),
                _p("Third key point", bullet=True, space_after=10),
            ]),
        ],
    },
    "two_col": {
        "name": "Two columns",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Slide title")]),
            _text(_ML, 130, 408, 340, "body", [
                _p("Left column heading", bold=True, color="dk2", bullet=False, space_after=10),
                _p("Left point one", bullet=True, space_after=6),
                _p("Left point two", bullet=True, space_after=6),
            ]),
            _text(504, 130, 408, 340, "body", [
                _p("Right column heading", bold=True, color="dk2", bullet=False, space_after=10),
                _p("Right point one", bullet=True, space_after=6),
                _p("Right point two", bullet=True, space_after=6),
            ]),
        ],
    },
    "image_right": {
        "name": "Image right",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Slide title")]),
            _text(_ML, 130, 408, 340, "body", [
                _p("Describe the visual", bullet=True, space_after=8),
                _p("Second supporting point", bullet=True, space_after=8),
            ]),
            {"type": "image", "x": 504, "y": 130, "w": 408, "h": 300, "token": "", "fit": "contain"},
        ],
    },
    "table": {
        "name": "Table",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Slide title")]),
            {"type": "table", "x": _ML, "y": 130, "w": _MW, "h": 300, "header": True, "banding": True,
             "col_widths": [288, 288, 288],
             "rows": [
                 [{"t": "Column A"}, {"t": "Column B"}, {"t": "Column C"}],
                 [{"t": "Row 1"}, {"t": "—"}, {"t": "—"}],
                 [{"t": "Row 2"}, {"t": "—"}, {"t": "—"}],
                 [{"t": "Row 3"}, {"t": "—"}, {"t": "—"}],
             ]},
        ],
    },
    "metric": {
        "name": "Metric",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Slide title")]),
            _text(_ML, 170, 430, 120, "headline", [_p("42%", size=68, bold=True, color="accent1")]),
            _text(_ML, 296, 430, 40, "subhead", [_p("What this number means")]),
            _text(504, 176, 408, 240, "body", [
                _p("Context for the metric", bullet=True, space_after=10),
                _p("Why it matters", bullet=True, space_after=10),
            ]),
        ],
    },
    "chart": {
        "name": "Chart",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Slide title")]),
            {"type": "chart", "x": _ML, "y": 130, "w": 520, "h": 320, "chart": "column",
             "title": "", "legend": True,
             "categories": ["Q1", "Q2", "Q3", "Q4"],
             "series": [{"name": "Series 1", "values": [3, 5, 4, 7]}]},
            _text(600, 150, 312, 280, "body", [
                _p("What the chart shows", bullet=True, space_after=10),
                _p("The takeaway", bullet=True, space_after=10),
            ]),
        ],
    },
    "quote": {
        "name": "Quote",
        "bg": "lt2",
        "elements": [
            _text(120, 180, 720, 200, "quote", [_p("“A short, memorable quotation that anchors the slide.”", align="center")], valign="middle"),
            _text(120, 380, 720, 30, "caption", [_p("— Attribution", align="center")]),
        ],
    },
    "closing": {
        "name": "Closing",
        "skip_footer": True,
        "bg": "dk2",
        "elements": [
            _text(_ML, 220, _MW, 80, "headline", [_p("Thank you", color="lt1", align="center")]),
            _text(_ML, 315, _MW, 40, "subhead", [_p("name@example.com", color="lt2", align="center")]),
        ],
    },
    "blank": {
        "name": "Blank",
        "elements": [],
    },
}

LAYOUT_IDS = tuple(_LAYOUTS.keys())


def get_layout(layout_id: str) -> dict | None:
    """Return a deep copy of a layout seed slide (no ids), or None if unknown."""
    seed = _LAYOUTS.get(layout_id)
    return copy.deepcopy(seed) if seed is not None else None


def layout_catalog() -> list[dict]:
    """``[{"id","name"}]`` for the skill/prompt to advertise the choices."""
    return [{"id": lid, "name": _LAYOUTS[lid]["name"]} for lid in LAYOUT_IDS]
