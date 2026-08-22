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
    "photo": {
        "name": "Photo cover",
        "skip_footer": True,
        "bg": "dk2",
        "bg_image": "",  # set a "[[image:UUID]]" token to fill the slide with a photo
        "bg_scrim": {"color": "dk1", "opacity": 0.45},
        "elements": [
            _text(_ML, 220, _MW, 90, "headline", [_p("Headline over a photo", color="lt1", align="center")], valign="bottom"),
            _text(_ML, 320, _MW, 40, "subhead", [_p("Subtitle", color="lt2", align="center")]),
        ],
    },
    "agenda": {
        "name": "Agenda",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Agenda")]),
            {"type": "shape", "x": _ML, "y": 150, "w": 42, "h": 42, "shape": "oval", "fill": "accent1",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "1", "color": "lt1", "b": True}]}]}},
            _text(112, 157, 740, 32, "subhead", [_p("First topic")]),
            {"type": "shape", "x": _ML, "y": 222, "w": 42, "h": 42, "shape": "oval", "fill": "accent1",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "2", "color": "lt1", "b": True}]}]}},
            _text(112, 229, 740, 32, "subhead", [_p("Second topic")]),
            {"type": "shape", "x": _ML, "y": 294, "w": 42, "h": 42, "shape": "oval", "fill": "accent1",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "3", "color": "lt1", "b": True}]}]}},
            _text(112, 301, 740, 32, "subhead", [_p("Third topic")]),
            {"type": "shape", "x": _ML, "y": 366, "w": 42, "h": 42, "shape": "oval", "fill": "accent1",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "4", "color": "lt1", "b": True}]}]}},
            _text(112, 373, 740, 32, "subhead", [_p("Fourth topic")]),
        ],
    },
    "exec_summary": {
        "name": "Executive summary",
        "elements": [
            _text(_ML, 44, _MW, 90, "headline", [_p("The bottom line — write the recommendation as the title")]),
            _text(_ML, 160, _MW, 300, "body", [
                _p("Key message one — lead with the answer, then the support.", bullet=True, space_after=14),
                _p("Key message two — quantify the impact where you can.", bullet=True, space_after=14),
                _p("Key message three — end with the recommended next step.", bullet=True, space_after=14),
            ]),
        ],
    },
    "kpi_row": {
        "name": "KPI row",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Key results")]),
            _text(_ML, 190, 264, 90, "headline", [_p("42%", size=54, bold=True, color="accent1", align="center")]),
            _text(_ML, 288, 264, 36, "subhead", [_p("Metric one", align="center")]),
            _text(348, 190, 264, 90, "headline", [_p("3.4x", size=54, bold=True, color="accent2", align="center")]),
            _text(348, 288, 264, 36, "subhead", [_p("Metric two", align="center")]),
            _text(648, 190, 264, 90, "headline", [_p("$6M", size=54, bold=True, color="accent4", align="center")]),
            _text(648, 288, 264, 36, "subhead", [_p("Metric three", align="center")]),
        ],
    },
    "process": {
        "name": "Process",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("How it works")]),
            {"type": "shape", "x": _ML, "y": 210, "w": 258, "h": 120, "shape": "chevron", "fill": "accent4",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Step 1", "color": "lt1", "b": True}]}]}},
            {"type": "shape", "x": 324, "y": 210, "w": 258, "h": 120, "shape": "chevron", "fill": "accent2",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Step 2", "color": "lt1", "b": True}]}]}},
            {"type": "shape", "x": 596, "y": 210, "w": 258, "h": 120, "shape": "chevron", "fill": "accent1",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Step 3", "color": "lt1", "b": True}]}]}},
        ],
    },
    "timeline": {
        "name": "Timeline",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Roadmap")]),
            {"type": "line", "x1": 90, "y1": 272, "x2": 872, "y2": 272, "color": "accent3", "w": 3},
            {"type": "shape", "x": 110, "y": 258, "w": 28, "h": 28, "shape": "oval", "fill": "accent1"},
            _text(44, 296, 190, 30, "data", [_p("Q1 · Milestone", align="center")]),
            {"type": "shape", "x": 360, "y": 258, "w": 28, "h": 28, "shape": "oval", "fill": "accent1"},
            _text(294, 296, 190, 30, "data", [_p("Q2 · Milestone", align="center")]),
            {"type": "shape", "x": 610, "y": 258, "w": 28, "h": 28, "shape": "oval", "fill": "accent1"},
            _text(544, 296, 190, 30, "data", [_p("Q3 · Milestone", align="center")]),
            {"type": "shape", "x": 830, "y": 258, "w": 28, "h": 28, "shape": "oval", "fill": "accent1"},
            _text(770, 296, 142, 30, "data", [_p("Q4 · Milestone", align="center")]),
        ],
    },
    "matrix_2x2": {
        "name": "2x2 matrix",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Prioritisation")]),
            {"type": "line", "x1": 480, "y1": 132, "x2": 480, "y2": 458, "color": "accent3", "w": 2},
            {"type": "line", "x1": 120, "y1": 295, "x2": 840, "y2": 295, "color": "accent3", "w": 2},
            _text(140, 150, 320, 30, "subhead", [_p("Quadrant A", align="center")]),
            _text(500, 150, 320, 30, "subhead", [_p("Quadrant B", align="center")]),
            _text(140, 410, 320, 30, "subhead", [_p("Quadrant C", align="center")]),
            _text(500, 410, 320, 30, "subhead", [_p("Quadrant D", align="center")]),
        ],
    },
    "comparison": {
        "name": "Comparison",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Comparison")]),
            {"type": "line", "x1": 480, "y1": 140, "x2": 480, "y2": 450, "color": "accent3", "w": 2},
            _text(_ML, 140, 400, 40, "subhead", [_p("Option A", bold=True, color="dk2")]),
            _text(_ML, 196, 400, 240, "body", [
                _p("First point", bullet=True, space_after=8),
                _p("Second point", bullet=True, space_after=8),
            ]),
            _text(504, 140, 400, 40, "subhead", [_p("Option B", bold=True, color="accent1")]),
            _text(504, 196, 400, 240, "body", [
                _p("First point", bullet=True, space_after=8),
                _p("Second point", bullet=True, space_after=8),
            ]),
        ],
    },
    "team": {
        "name": "Team",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Team")]),
            {"type": "image", "x": 90, "y": 150, "w": 180, "h": 180, "token": "", "fit": "cover"},
            _text(60, 342, 240, 30, "subhead", [_p("Name", align="center")]),
            _text(60, 376, 240, 26, "caption", [_p("Role", align="center")]),
            {"type": "image", "x": 390, "y": 150, "w": 180, "h": 180, "token": "", "fit": "cover"},
            _text(360, 342, 240, 30, "subhead", [_p("Name", align="center")]),
            _text(360, 376, 240, 26, "caption", [_p("Role", align="center")]),
            {"type": "image", "x": 690, "y": 150, "w": 180, "h": 180, "token": "", "fit": "cover"},
            _text(660, 342, 240, 30, "subhead", [_p("Name", align="center")]),
            _text(660, 376, 240, 26, "caption", [_p("Role", align="center")]),
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
