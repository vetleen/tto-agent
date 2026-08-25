"""Seed slide JSON for ``slides_add_slide(layout=...)``.

Each entry is a ready-to-edit single slide on the 960x540 grid (48pt side
margins -> 864pt content width). The tool deep-copies the seed, mints fresh
ids, and inserts it; the model then edits the placeholder text freely. Seeds use
theme text CLASSES and colour NAMES, never literals, so they inherit branding.

A seed's ``description`` is catalogue-only (stripped on insert); a ``comment``
is a Slide schema field, so it travels with the inserted slide as an authoring
note for the model.
"""

from __future__ import annotations

import copy

# Shared grid constants (points).
_ML = 48          # left margin
_MW = 864         # content width (960 - 2*48)
_MR = 912         # right edge of content
# Two-column split: 2 * _COL + 48pt gutter == _MW, so col 1 spans 48..456 and
# col 2 spans 504..912.
_COL = 408        # column width
_C2X = 504        # x of the second column

# Golden-ratio split, for the layouts that divide the slide into two unequal
# halves (a colour panel + content, a numeral + a heading, text + a bleeding
# image). One shared proportion keeps them reading as a set; WHICH side is the
# small one varies per layout.
#
#   960 / φ² == 366.7 -> the minor side is 366pt, the major 594pt.
#
# Both sides carry the usual 48pt inner padding, so:
#   minor on the left  -> minor text 48..318 (_SPLIT_W), major text 414..912
#   minor on the right -> major text 48..546,            minor text 642..912
_SPLIT = 366      # x of the split line when the LEFT side is the small one
_SPLIT_W = 270    # text width inside the minor side (366 - 2*48)
_SPLIT_X = 414    # x of the text on the major side (366 + 48)
_SPLIT_MW = 498   # text width on the major side (912 - 414)


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
    "title_01": {
        "name": "Title 1",
        "description": "Editorial cover — a big left-aligned title with a subtitle underneath, nothing else.",
        "skip_footer": True,
        "bg": "lt1",
        "elements": [
            _text(_ML, 190, 728, 130, "headline", [_p("Presentation title", size=54)], valign="bottom"),
            _text(_ML, 330, 728, 40, "subhead", [_p("Subtitle or author · date", size=22)]),
        ],
    },
    "title_02": {
        "name": "Title 2",
        "description": "Editorial cover — a big left-aligned title with a company-logo slot above it and a hairline footer band carrying date, author and contact.",
        "comment": "Drop the logo image or any of the three footer slots that don't apply, but keep the hairline — it's what anchors the band. For a photo cover use the 'Photo cover' layout instead.",
        "skip_footer": True,
        "bg": "lt1",
        "elements": [
            {"type": "image", "x": _ML, "y": 40, "w": 100, "h": 40, "token": "[[image:company-logo]]", "fit": "contain"},
            _text(_ML, 170, 728, 130, "headline", [_p("Presentation title", size=54)], valign="bottom"),
            _text(_ML, 312, 728, 40, "subhead", [_p("Subtitle or tagline", size=22)]),
            {"type": "line", "x1": _ML, "y1": 462, "x2": _MR, "y2": 462, "color": "accent3", "w": 1},
            _text(_ML, 476, 280, 26, "data", [_p("Date")]),
            _text(340, 476, 280, 26, "data", [_p("Author", align="center")]),
            _text(632, 476, 280, 26, "data", [_p("presenter@example.com", align="right")]),
        ],
    },
    "title_03": {
        "name": "Title 3",
        "description": "Split cover — title, subtitle and author on a full-height colour panel, with a cover image bleeding off the other half.",
        "comment": "Replace the image token with a real cover photo (fit=cover crops it to the panel). To mirror the layout, move the panel and its text to the right (x=594) and the image to x=0.",
        "skip_footer": True,
        "bg": "lt1",
        "elements": [
            {"type": "shape", "x": 0, "y": 0, "w": _SPLIT, "h": 540, "shape": "rect", "fill": "dk2"},
            {"type": "image", "x": _SPLIT, "y": 0, "w": 960 - _SPLIT, "h": 540, "token": "[[image:cover-image]]", "fit": "cover"},
            {"type": "image", "x": _ML, "y": 40, "w": 100, "h": 40, "token": "[[image:company-logo]]", "fit": "contain"},
            _text(_ML, 250, _SPLIT_W, 160, "headline", [_p("Presentation title", size=36, color="lt1")], valign="bottom"),
            _text(_ML, 420, _SPLIT_W, 34, "subhead", [_p("Subtitle or tagline", size=18)]),
            _text(_ML, 464, _SPLIT_W, 26, "data", [_p("Author · date", color="lt2")]),
        ],
    },
    "section": {
        "name": "Section",
        "description": "Section divider — a large centred heading on a dark background to break the deck into parts.",
        "bg": "dk2",
        "elements": [
            _text(_ML, 210, _MW, 90, "headline", [_p("Section heading", color="lt1", align="center")], valign="bottom"),
            {"type": "line", "x1": 420, "y1": 320, "x2": 540, "y2": 320, "color": "accent1", "w": 3},
        ],
    },
    "section_02": {
        "name": "Section 2",
        "description": "Section divider — a giant section number beside the heading and a one-line sub-heading, split by a vertical hairline.",
        "bg": "dk2",
        "elements": [
            _text(_ML, 145, _SPLIT_W, 250, "headline", [_p("01", size=180, color="accent1", align="center")], valign="middle"),
            {"type": "line", "x1": _SPLIT, "y1": 135, "x2": _SPLIT, "y2": 405, "color": "lt2", "w": 1},
            _text(_SPLIT_X, 200, _SPLIT_MW, 130, "headline", [_p("Section heading", size=44, color="lt1")], valign="bottom"),
            _text(_SPLIT_X, 344, _SPLIT_MW, 56, "body", [_p("One line on what this section covers", size=16, color="lt2")]),
        ],
    },
    "bullets": {
        "name": "Bullets",
        "description": "The workhorse content slide — a title over a simple bulleted list (levels 0–2 get ‣ / – / ◦ markers).",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Slide title")]),
            _text(_ML, 130, _MW, 340, "body", [
                _p("First key point", bullet=True, space_after=10),
                _p("A supporting sub-point", bullet=True, level=1, space_after=10),
                _p("Second key point", bullet=True, space_after=10),
                _p("Third key point", bullet=True, space_after=10),
            ]),
        ],
    },
    "two_col": {
        "name": "Two columns",
        "description": "A title over two even side-by-side columns, each opened by a rule and its own column heading.",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Slide title")]),
            {"type": "line", "x1": _ML, "y1": 132, "x2": _ML + _COL, "y2": 132, "color": "dk2", "w": 1},
            _text(_ML, 142, _COL, 34, "headline", [_p("First column", size=20)]),
            _text(_ML, 182, _COL, 280, "body", [
                _p("First point", bullet=True, size=13, space_after=8),
                _p("Second point", bullet=True, size=13, space_after=8),
                _p("Third point", bullet=True, size=13, space_after=8),
            ]),
            {"type": "line", "x1": _C2X, "y1": 132, "x2": _MR, "y2": 132, "color": "dk2", "w": 1},
            _text(_C2X, 142, _COL, 34, "headline", [_p("Second column", size=20)]),
            _text(_C2X, 182, _COL, 280, "body", [
                _p("First point", bullet=True, size=13, space_after=8),
                _p("Second point", bullet=True, size=13, space_after=8),
                _p("Third point", bullet=True, size=13, space_after=8),
            ]),
        ],
    },
    "three_column": {
        "name": "Three columns",
        "description": "A title over three even ruled columns — a heading and a short bulleted list in each.",
        "comment": "Three 264pt columns on 36pt gutters. For two columns use the 'Two columns' layout, which keeps the same rule-and-heading treatment on a wider grid.",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Slide title")]),
            {"type": "line", "x1": _ML, "y1": 132, "x2": 312, "y2": 132, "color": "dk2", "w": 1},
            _text(_ML, 142, 264, 34, "headline", [_p("First column", size=20)]),
            _text(_ML, 182, 264, 280, "body", [
                _p("First point", bullet=True, size=13, space_after=8),
                _p("Second point", bullet=True, size=13, space_after=8),
                _p("Third point", bullet=True, size=13, space_after=8),
            ]),
            {"type": "line", "x1": 348, "y1": 132, "x2": 612, "y2": 132, "color": "dk2", "w": 1},
            _text(348, 142, 264, 34, "headline", [_p("Second column", size=20)]),
            _text(348, 182, 264, 280, "body", [
                _p("First point", bullet=True, size=13, space_after=8),
                _p("Second point", bullet=True, size=13, space_after=8),
                _p("Third point", bullet=True, size=13, space_after=8),
            ]),
            {"type": "line", "x1": 648, "y1": 132, "x2": _MR, "y2": 132, "color": "dk2", "w": 1},
            _text(648, 142, 264, 34, "headline", [_p("Third column", size=20)]),
            _text(648, 182, 264, 280, "body", [
                _p("First point", bullet=True, size=13, space_after=8),
                _p("Second point", bullet=True, size=13, space_after=8),
                _p("Third point", bullet=True, size=13, space_after=8),
            ]),
        ],
    },
    "image_right": {
        "name": "Image right",
        "description": "Bullets on the left, an image on the right.",
        "comment": "The image can sit on either side: swap the x of the text column (48) and the image (504) to flip the layout.",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Slide title")]),
            _text(_ML, 130, _COL, 340, "body", [
                _p("Describe the visual", bullet=True, space_after=8),
                _p("Second supporting point", bullet=True, space_after=8),
            ]),
            {"type": "image", "x": _C2X, "y": 130, "w": _COL, "h": 300, "token": "", "fit": "contain"},
        ],
    },
    "image_bleed": {
        "name": "Image half-bleed",
        "description": "Title + bullets on the left, an image bleeding off the right half of the slide (full height, edge to edge).",
        "comment": "The image takes the larger side edge-to-edge (fit=cover crops it). To mirror the layout, put the image at x=0 and move the text column to x=642.",
        "skip_footer": True,
        "elements": [
            {"type": "image", "x": _SPLIT, "y": 0, "w": 960 - _SPLIT, "h": 540, "token": "", "fit": "cover"},
            _text(_ML, 44, _SPLIT_W, 90, "headline", [_p("Slide title")]),
            _text(_ML, 150, _SPLIT_W, 320, "body", [
                _p("Describe the visual", bullet=True, space_after=8),
                _p("Second supporting point", bullet=True, space_after=8),
            ]),
        ],
    },
    "table": {
        "name": "Table",
        "description": "A title over a data table (header row + banded rows; col_widths sets relative column widths).",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Slide title")]),
            {"type": "table", "x": _ML, "y": 130, "w": _MW, "h": 300, "header": True, "banding": True,
             "col_widths": [312, 276, 276],
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
        "description": "One big hero number on a full-height colour panel, with the supporting bullets beside it.",
        "elements": [
            {"type": "shape", "x": 0, "y": 0, "w": _SPLIT, "h": 540, "shape": "rect", "fill": "dk2"},
            _text(_ML, 44, _SPLIT_W, 60, "headline", [_p("Key results", color="lt1")]),
            _text(_ML, 180, _SPLIT_W, 160, "headline", [_p("42%", size=120, bold=True, color="accent1")], valign="bottom"),
            _text(_ML, 356, _SPLIT_W, 40, "subhead", [_p("What this number means", size=18, color="lt1")]),
            _text(_SPLIT_X, 186, _SPLIT_MW, 260, "body", [
                _p("Context for the metric", bullet=True, space_after=10),
                _p("Why it matters", bullet=True, space_after=10),
                _p("What we do next", bullet=True, space_after=10),
            ]),
        ],
    },
    "chart": {
        "name": "Chart",
        "description": "A chart on the left with takeaway bullets on the right.",
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
        "description": "Full-bleed photo cover — a big headline anchored bottom-left over a background image with a dark scrim.",
        "skip_footer": True,
        "bg": "dk2",
        "bg_image": "",  # set a "[[image:UUID]]" token to fill the slide with a photo
        "bg_scrim": {"color": "dk1", "opacity": 0.45},
        "elements": [
            _text(_ML, 330, 700, 110, "headline", [_p("Headline over a photo", size=52, color="lt1")], valign="bottom"),
            _text(_ML, 456, 640, 40, "body", [_p("One line that sets up the story", size=16, color="lt2")]),
        ],
    },
    "agenda": {
        "name": "Agenda",
        "description": "A numbered agenda / contents list — big editorial numerals, a topic and a one-line gloss per row, with an optional duration on the right.",
        "comment": "Four 88pt rows from y=144, separated by hairlines. Drop a row (numeral + topic + gloss + duration, and the hairline above it) for a shorter agenda; drop the right-hand duration column if timings aren't relevant.",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Agenda")]),
            _text(_ML, 144, 96, 52, "headline", [_p("01", size=40, color="accent3")]),
            _text(168, 142, 520, 34, "headline", [_p("First topic", size=24)]),
            _text(168, 182, 520, 28, "body", [_p("One line on what this part covers", size=13)]),
            _text(760, 150, 152, 24, "caption", [_p("10 min", size=12, align="right")]),
            {"type": "line", "x1": _ML, "y1": 222, "x2": _MR, "y2": 222, "color": "accent3", "w": 1},
            _text(_ML, 232, 96, 52, "headline", [_p("02", size=40, color="accent3")]),
            _text(168, 230, 520, 34, "headline", [_p("Second topic", size=24)]),
            _text(168, 270, 520, 28, "body", [_p("One line on what this part covers", size=13)]),
            _text(760, 238, 152, 24, "caption", [_p("15 min", size=12, align="right")]),
            {"type": "line", "x1": _ML, "y1": 310, "x2": _MR, "y2": 310, "color": "accent3", "w": 1},
            _text(_ML, 320, 96, 52, "headline", [_p("03", size=40, color="accent3")]),
            _text(168, 318, 520, 34, "headline", [_p("Third topic", size=24)]),
            _text(168, 358, 520, 28, "body", [_p("One line on what this part covers", size=13)]),
            _text(760, 326, 152, 24, "caption", [_p("20 min", size=12, align="right")]),
            {"type": "line", "x1": _ML, "y1": 398, "x2": _MR, "y2": 398, "color": "accent3", "w": 1},
            _text(_ML, 408, 96, 52, "headline", [_p("04", size=40, color="accent3")]),
            _text(168, 406, 520, 34, "headline", [_p("Fourth topic", size=24)]),
            _text(168, 446, 520, 28, "body", [_p("One line on what this part covers", size=13)]),
            _text(760, 414, 152, 24, "caption", [_p("5 min", size=12, align="right")]),
        ],
    },
    "exec_summary": {
        "name": "Executive summary",
        "description": "The bottom line — an action-title recommendation on the left, three numbered supporting messages stacked on the right.",
        "elements": [
            _text(_ML, 88, _SPLIT_W, 260, "headline", [_p("The bottom line — write the recommendation as the title", size=34)]),
            _text(_SPLIT_X, 88, _SPLIT_MW, 30, "subhead", [_p("1")]),
            _text(_SPLIT_X, 120, _SPLIT_MW, 70, "body", [_p("Key message one — lead with the answer, then the support.")]),
            {"type": "line", "x1": _SPLIT_X, "y1": 200, "x2": _MR, "y2": 200, "color": "accent3", "w": 1},
            _text(_SPLIT_X, 216, _SPLIT_MW, 30, "subhead", [_p("2")]),
            _text(_SPLIT_X, 248, _SPLIT_MW, 70, "body", [_p("Key message two — quantify the impact where you can.")]),
            {"type": "line", "x1": _SPLIT_X, "y1": 328, "x2": _MR, "y2": 328, "color": "accent3", "w": 1},
            _text(_SPLIT_X, 344, _SPLIT_MW, 30, "subhead", [_p("3")]),
            _text(_SPLIT_X, 376, _SPLIT_MW, 70, "body", [_p("Key message three — end with the recommended next step.")]),
        ],
    },
    "kpi_row": {
        "name": "KPI row",
        "description": "Three KPIs side by side — a big number, its label and one line of context each, split by vertical hairlines.",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Key results")]),
            # Three 264pt columns (36pt gutters) with the hairline centred in each
            # gutter — the same grid the executive summary uses.
            _text(_ML, 180, 264, 92, "headline", [_p("42%", size=62, bold=True, color="accent1")]),
            _text(_ML, 286, 264, 28, "body", [_p("Metric one", bold=True, size=16, color="dk2")]),
            _text(_ML, 322, 264, 90, "caption", [_p("One line of context on why this number matters.", size=12)]),
            {"type": "line", "x1": 330, "y1": 180, "x2": 330, "y2": 440, "color": "accent3", "w": 1},
            _text(348, 180, 264, 92, "headline", [_p("3.4x", size=62, bold=True, color="accent1")]),
            _text(348, 286, 264, 28, "body", [_p("Metric two", bold=True, size=16, color="dk2")]),
            _text(348, 322, 264, 90, "caption", [_p("One line of context on why this number matters.", size=12)]),
            {"type": "line", "x1": 630, "y1": 180, "x2": 630, "y2": 440, "color": "accent3", "w": 1},
            _text(648, 180, 264, 92, "headline", [_p("$6M", size=62, bold=True, color="accent1")]),
            _text(648, 286, 264, 28, "body", [_p("Metric three", bold=True, size=16, color="dk2")]),
            _text(648, 322, 264, 90, "caption", [_p("One line of context on why this number matters.", size=12)]),
        ],
    },
    "process": {
        "name": "Process",
        "description": "A left-to-right process flow of 4 chevron steps, each with a line of detail underneath.",
        "comment": "Drop a chevron and its caption for a 3-step flow (then widen the rest to keep them flush with both margins).",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("How it works")]),
            # 4 × 207pt chevrons + 3 × 12pt gaps == 864pt, flush with both margins.
            # The captions inset 8pt to clear each chevron's left notch.
            {"type": "shape", "x": _ML, "y": 176, "w": 207, "h": 96, "shape": "chevron", "fill": "accent4",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Step 1", "color": "lt1", "b": True}]}]}},
            {"type": "shape", "x": 267, "y": 176, "w": 207, "h": 96, "shape": "chevron", "fill": "accent2",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Step 2", "color": "lt1", "b": True}]}]}},
            {"type": "shape", "x": 486, "y": 176, "w": 207, "h": 96, "shape": "chevron", "fill": "accent1",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Step 3", "color": "lt1", "b": True}]}]}},
            {"type": "shape", "x": 705, "y": 176, "w": 207, "h": 96, "shape": "chevron", "fill": "accent6",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Step 4", "color": "lt1", "b": True}]}]}},
            _text(56, 292, 191, 90, "body", [_p("One line on what happens in this step.", size=13)]),
            _text(275, 292, 191, 90, "body", [_p("One line on what happens in this step.", size=13)]),
            _text(494, 292, 191, 90, "body", [_p("One line on what happens in this step.", size=13)]),
            _text(713, 292, 191, 90, "body", [_p("One line on what happens in this step.", size=13)]),
        ],
    },
    "cycle": {
        "name": "Cycle / flywheel",
        "description": "A cyclical / flywheel diagram — four stages marked around a ring with the loop's name at its centre.",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("The growth cycle")]),
            # A 256pt ring centred on (480, 307); the four station dots sit ON the
            # ring at its N/E/S/W points, each label reading outward from it.
            {"type": "shape", "x": 352, "y": 179, "w": 256, "h": 256, "shape": "oval",
             "line": {"color": "accent3", "w": 2}},
            _text(380, 294, 200, 30, "subhead", [_p("GROWTH", size=20, align="center")]),
            {"type": "shape", "x": 470, "y": 169, "w": 20, "h": 20, "shape": "oval", "fill": "dk2"},
            _text(380, 112, 200, 30, "subhead", [_p("Stage 1", size=18, color="dk2", align="center")], valign="bottom"),
            {"type": "shape", "x": 598, "y": 297, "w": 20, "h": 20, "shape": "oval", "fill": "dk2"},
            _text(632, 292, 190, 30, "subhead", [_p("Stage 2", size=18, color="dk2")], valign="middle"),
            {"type": "shape", "x": 470, "y": 425, "w": 20, "h": 20, "shape": "oval", "fill": "dk2"},
            _text(380, 456, 200, 30, "subhead", [_p("Stage 3", size=18, color="dk2", align="center")]),
            {"type": "shape", "x": 342, "y": 297, "w": 20, "h": 20, "shape": "oval", "fill": "dk2"},
            _text(138, 292, 190, 30, "subhead", [_p("Stage 4", size=18, color="dk2", align="right")], valign="middle"),
        ],
    },
    "timeline": {
        "name": "Timeline",
        "description": "A horizontal timeline / roadmap — milestone dots on a line, each with a label and a line of detail beneath it.",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Roadmap")]),
            # One line across the content width; a dot sits ON it at the start of
            # each 216pt column, with the milestone text hanging left-aligned below.
            {"type": "line", "x1": _ML, "y1": 256, "x2": _MR, "y2": 256, "color": "accent3", "w": 2},
            {"type": "shape", "x": _ML, "y": 246, "w": 20, "h": 20, "shape": "oval", "fill": "accent1"},
            _text(_ML, 292, 180, 30, "subhead", [_p("Q1 · Milestone", size=20, color="dk2")]),
            _text(_ML, 328, 180, 70, "caption", [_p("One line on what lands here.", size=12)]),
            {"type": "shape", "x": 264, "y": 246, "w": 20, "h": 20, "shape": "oval", "fill": "accent1"},
            _text(264, 292, 180, 30, "subhead", [_p("Q2 · Milestone", size=20, color="dk2")]),
            _text(264, 328, 180, 70, "caption", [_p("One line on what lands here.", size=12)]),
            {"type": "shape", "x": 480, "y": 246, "w": 20, "h": 20, "shape": "oval", "fill": "accent1"},
            _text(480, 292, 180, 30, "subhead", [_p("Q3 · Milestone", size=20, color="dk2")]),
            _text(480, 328, 180, 70, "caption", [_p("One line on what lands here.", size=12)]),
            {"type": "shape", "x": 696, "y": 246, "w": 20, "h": 20, "shape": "oval", "fill": "accent1"},
            _text(696, 292, 180, 30, "subhead", [_p("Q4 · Milestone", size=20, color="dk2")]),
            _text(696, 328, 180, 70, "caption", [_p("One line on what lands here.", size=12)]),
        ],
    },
    "matrix_2x2": {
        "name": "2x2 matrix",
        "description": "A 2×2 matrix of filled quadrant tiles with labelled axes — fill the quadrant that carries the recommendation in the accent colour.",
        "comment": "Quadrant B is the highlighted tile (accent1 fill, light text); the other three are pale panels with dark text. Move the highlight by swapping the fills and the text colours.",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Prioritisation")]),
            # Four 294×148 tiles on a 12pt column gap / 8pt row gap.
            {"type": "shape", "x": 260, "y": 140, "w": 294, "h": 148, "shape": "rect", "fill": "lt2"},
            _text(280, 156, 254, 28, "body", [_p("Quadrant A", bold=True, size=15, color="dk2")]),
            _text(280, 184, 254, 60, "body", [_p("One line on what sits here.", size=12)]),
            {"type": "shape", "x": 566, "y": 140, "w": 294, "h": 148, "shape": "rect", "fill": "accent1"},
            _text(586, 156, 254, 28, "body", [_p("Quadrant B — do first", bold=True, size=15, color="lt1")]),
            _text(586, 184, 254, 60, "body", [_p("One line on what sits here.", size=12, color="lt1")]),
            {"type": "shape", "x": 260, "y": 296, "w": 294, "h": 148, "shape": "rect", "fill": "lt2"},
            _text(280, 312, 254, 28, "body", [_p("Quadrant C", bold=True, size=15, color="dk2")]),
            _text(280, 340, 254, 60, "body", [_p("One line on what sits here.", size=12)]),
            {"type": "shape", "x": 566, "y": 296, "w": 294, "h": 148, "shape": "rect", "fill": "lt2"},
            _text(586, 312, 254, 28, "body", [_p("Quadrant D", bold=True, size=15, color="dk2")]),
            _text(586, 340, 254, 60, "body", [_p("One line on what sits here.", size=12)]),
            _text(260, 456, 600, 24, "body", [_p("Effort →", bold=True, size=12, align="center")]),
            _text(150, 284, 140, 24, "body", [_p("Impact →", bold=True, size=12, align="center")], rotation=-90),
        ],
    },
    "ecosystem": {
        "name": "Ecosystem / network",
        "description": "An ecosystem / network map — a central hub connected to surrounding partners.",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Our ecosystem")]),
            {
                "type": "network", "x": 210, "y": 150, "w": 560, "h": 330,
                "node_color": "accent2", "edge_color": "accent2", "node_r": 6, "label_size": 13,
                "nodes": [
                    {"x": 270, "y": 150, "label": "Us", "label_pos": "c", "emphasis": True,
                     "r": 0, "size": 26, "color": "dk1"},
                    {"x": 270, "y": 18, "label": "Partner A", "label_pos": "t"},
                    {"x": 500, "y": 70, "label": "Partner B", "label_pos": "r"},
                    {"x": 520, "y": 240, "label": "Partner C", "label_pos": "r"},
                    {"x": 270, "y": 300, "label": "Partner D", "label_pos": "b"},
                    {"x": 40, "y": 240, "label": "Partner E", "label_pos": "l"},
                    {"x": 20, "y": 70, "label": "Partner F", "label_pos": "l"},
                ],
                "edges": [
                    {"a": 0, "b": 1}, {"a": 0, "b": 2}, {"a": 0, "b": 3},
                    {"a": 0, "b": 4}, {"a": 0, "b": 5}, {"a": 0, "b": 6},
                    {"a": 1, "b": 2}, {"a": 2, "b": 3}, {"a": 3, "b": 4},
                    {"a": 4, "b": 5}, {"a": 5, "b": 6}, {"a": 6, "b": 1},
                ],
            },
        ],
    },
    "comparison": {
        "name": "Comparison",
        "description": "A side-by-side comparison of two options (A vs B).",
        "elements": [
            _text(_ML, 44, _MW, 60, "headline", [_p("Comparison")]),
            {"type": "line", "x1": 480, "y1": 140, "x2": 480, "y2": 450, "color": "accent3", "w": 2},
            _text(_ML, 140, _COL, 40, "subhead", [_p("Option A", bold=True, color="dk2")]),
            _text(_ML, 196, _COL, 240, "body", [
                _p("First point", bullet=True, space_after=8),
                _p("Second point", bullet=True, space_after=8),
            ]),
            _text(_C2X, 140, _COL, 40, "subhead", [_p("Option B", bold=True, color="accent1")]),
            _text(_C2X, 196, _COL, 240, "body", [
                _p("First point", bullet=True, space_after=8),
                _p("Second point", bullet=True, space_after=8),
            ]),
        ],
    },
    "quote": {
        "name": "Quote",
        "description": "A large pull-quote with attribution.",
        "bg": "lt2",
        "elements": [
            _text(96, 70, 200, 160, "headline", [_p("“", size=150, bold=True, color="accent3", align="left")]),
            _text(120, 200, 720, 170, "quote", [_p("A short, memorable quotation that anchors the slide.", align="center")], valign="middle"),
            _text(120, 384, 720, 30, "caption", [_p("— Attribution", align="center")]),
        ],
    },
    "closing": {
        "name": "Closing",
        "description": "A closing / thank-you slide with contact details on a dark background.",
        "skip_footer": True,
        "bg": "dk2",
        "elements": [
            _text(_ML, 220, _MW, 80, "headline", [_p("Thank you", color="lt1", align="center")]),
            _text(_ML, 315, _MW, 40, "subhead", [_p("name@example.com", color="lt2", align="center")]),
        ],
    },
    "blank": {
        "name": "Blank",
        "description": "An empty slide to build from scratch.",
        "elements": [],
    },
}

LAYOUT_IDS = tuple(_LAYOUTS.keys())


def get_layout(layout_id: str) -> dict | None:
    """Return a deep copy of a layout seed slide (no ids), or None if unknown.

    ``description`` is a catalogue-only field (see ``layout_catalog``) — it is
    stripped here so the returned dict is a valid Slide (the schema forbids
    unknown keys). A seed's ``comment`` is a real Slide field and stays."""
    seed = _LAYOUTS.get(layout_id)
    if seed is None:
        return None
    seed = copy.deepcopy(seed)
    seed.pop("description", None)
    return seed


def layout_catalog() -> list[dict]:
    """``[{"id","name","description"}]`` for the slides_add_slide tool to advertise
    the choices so the model can pick the right layout confidently."""
    return [
        {"id": lid, "name": _LAYOUTS[lid]["name"], "description": _LAYOUTS[lid].get("description", "")}
        for lid in LAYOUT_IDS
    ]
