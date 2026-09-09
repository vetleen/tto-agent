"""Seed slide JSON for ``slides_add_slide(layout=...)`` and ``slides_create_deck(layouts=[...])``.

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

# ── 12-column grid (points) ─────────────────────────────────────────────────
# The 960x540 canvas carries 48pt side margins -> 864pt of content, divided into
# 12 columns on a 72pt pitch. Column c (1-indexed) STARTS at COL_X[c]; a column
# is 72pt wide with a 12pt gutter carved off its right, so a single column of
# content is 60pt and any block reaching column 12 runs flush to the 912pt right
# margin. `_span(a, b)` returns the (x, w) of a block covering columns a..b.
#
# These exact numbers are mirrored in the Slide Deck Collaborator skill (the
# model is handed the table, not the formula) — keep the two in sync.
_MARGIN = 48      # left content margin
_PITCH = 72       # column pitch (864 / 12)
_GUTTER = 12      # gap carved off the right of each column
_RIGHT = 912      # right content margin (48 + 12*72)

# Handy aliases for full-width elements (many seeds span the whole content box).
_ML = _MARGIN     # left edge of content   (48)
_MR = _RIGHT      # right edge of content  (912)
_MW = _RIGHT - _MARGIN  # full content width (864)

# 1-indexed column starts: COL_X[1]..COL_X[12]; COL_X[0] is padding so the
# columns read 1-based (COL_X[1] is column 1).
COL_X = [None] + [_MARGIN + (c - 1) * _PITCH for c in range(1, 13)]


def _span(a, b):
    """(x, w) for a block spanning 1-indexed columns ``a``..``b`` inclusive.

    A block whose right edge is column 12 runs flush to the right margin;
    otherwise it stops one 12pt gutter short of the next column's start."""
    x = COL_X[a]
    right = _RIGHT if b >= 12 else COL_X[b] + _PITCH - _GUTTER
    return x, right - x


def footer_section_box(start_col: int, colspan: int) -> tuple[int, int]:
    """(x, w) for a footer section beginning at 1-indexed ``start_col`` and spanning
    ``colspan`` grid columns (clamped to the 12-column grid). Shared by the pptx and
    pillow footer stampers so footer content lands on the same grid as slide bodies."""
    start_col = max(1, min(12, int(start_col)))
    end_col = max(start_col, min(12, start_col + int(colspan) - 1))
    return _span(start_col, end_col)


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


def _gtext(a, b, y, h, cls, paragraphs, **extra):
    """A text element snapped to grid columns ``a``..``b`` (see ``_span``)."""
    x, w = _span(a, b)
    return _text(x, y, w, h, cls, paragraphs, **extra)


_LAYOUTS: dict[str, dict] = {
    "title": {
        "name": "Title",
        "description": "A simple cover — a big left-aligned title with a subtitle underneath, nothing else; optional extras (company-logo slot, a hairline band with date / author / contact) are pre-designed in the slide comment. Use as the default title slide.",
        "comment": "Optional extras, pre-designed for this cover — add any that apply: (1) a company-logo slot above the title: image x=48 y=40 w=100 h=40 token [[image:company-logo]] fit contain (only if the theme or the user has a logo); (2) a hairline info band near the bottom: line x1=48 y1=462 x2=912 y2=462 color accent3 w=1, plus up to three `data` text elements at y=476 h=26 — date on cols 1-4 (x=48 w=276, left), author centred on cols 5-8 (x=336 w=276), contact right-aligned on cols 9-12 (x=624 w=288); keep the hairline whenever you use any of the three. The cover skips the stamped footer (set skip_footer false to show it — but not together with the info band). For a photo cover use the 'Photo cover' or 'Split cover' layout instead.",
        "skip_footer": True,
        "bg": "lt1",
        "elements": [
            _gtext(1, 10, 190, 130, "headline", [_p("Presentation title", size=54)], valign="bottom"),
            _gtext(1, 10, 330, 40, "subhead", [_p("Subtitle or author · date", size=22)]),
        ],
    },
    "title_split": {
        "name": "Split cover",
        "description": "Split cover — title, subtitle and author on a full-height colour panel, with a cover image bleeding off the other half. Use when you have (or want) an illustrative picture on the front page.",
        "comment": "Replace the image token with a real cover photo (fit=cover crops it to the panel). To mirror the layout, move the panel and its text to the right (panel x=552) and the image to x=0.",
        "skip_footer": True,
        "bg": "lt1",
        "elements": [
            # Colour panel fills columns 1-5 (x 0..408, bleeding off the left edge);
            # the cover image takes the rest. Panel text sits on columns 1-4.
            {"type": "shape", "x": 0, "y": 0, "w": 408, "h": 540, "shape": "rect", "fill": "dk2"},
            {"type": "image", "x": 408, "y": 0, "w": 552, "h": 540, "token": "[[image:cover-image]]", "fit": "cover"},
            {"type": "image", "x": _ML, "y": 40, "w": 100, "h": 40, "token": "[[image:company-logo]]", "fit": "contain"},
            _gtext(1, 4, 250, 160, "headline", [_p("Presentation title", size=36, color="lt1")], valign="bottom"),
            _gtext(1, 4, 420, 34, "subhead", [_p("Subtitle or tagline", size=18)]),
            _gtext(1, 4, 464, 26, "data", [_p("Author · date", color="lt2")]),
        ],
    },
    "section": {
        "name": "Section",
        "description": "Section divider — a large centred heading on a dark background, with an optional section number above it and a one-line sub-heading below (pre-designed in the slide comment). Use to open each part of a longer deck.",
        "comment": "Optional extras, pre-designed for this divider — add either or both: (1) a section number above the heading: `headline` text x=48 y=186 w=864 h=52 valign bottom, run '01' size 40 color accent1 align center (if the heading wraps to two lines, move it up to y=146); (2) a one-line sub-heading under the rule: `body` text x=120 y=336 w=708 h=30, run size 16 color lt2 align center. Keep the heading to one line where you can.",
        "bg": "dk2",
        "elements": [
            _gtext(1, 12, 210, 90, "headline", [_p("Section heading", color="lt1", align="center")], valign="bottom"),
            {"type": "line", "x1": 420, "y1": 320, "x2": 540, "y2": 320, "color": "accent1", "w": 3},
        ],
    },
    "bullets": {
        "name": "Bullets",
        "description": "The workhorse content slide — a title over a simple bulleted list with levels. Use for a plain list of points, or as the base slide for a table or a diagram recipe from the skill.",
        "elements": [
            _gtext(1, 12, 44, 60, "headline", [_p("Slide title")]),
            _gtext(1, 12, 130, 340, "body", [
                _p("First key point", bullet=True, space_after=10),
                _p("A supporting sub-point", bullet=True, level=1, space_after=10),
                _p("Second key point", bullet=True, space_after=10),
                _p("Third key point", bullet=True, space_after=10),
            ]),
        ],
    },
    "two_col": {
        "name": "Two columns",
        "description": "A title over two even side-by-side columns, each opened by a rule and its own column heading; fill the columns with whatever elements suit the point. Use for two parallel topics, a before/after, or an A-vs-B comparison (variant in the slide comment).",
        "comment": "The content per column is just a placeholder — replace it with non-text elements as you see fit. If you keep the column headings and rules, give both the same colour unless a colour difference is meant to draw attention. Variant — an A-vs-B comparison: replace each column's rule + heading with a `subhead` option name (y=140 h=40; the recommended option in accent1, the other dk2 — both dk2 when the comparison is neutral), start the bullets at y=196, and add a vertical hairline between the columns: line x1=474 y1=140 x2=474 y2=450 color accent3 w=2.",
        "elements": [
            _gtext(1, 12, 44, 60, "headline", [_p("Slide title")]),
            {"type": "line", "x1": 48, "y1": 132, "x2": 468, "y2": 132, "color": "dk2", "w": 1},
            _gtext(1, 6, 142, 34, "headline", [_p("First column", size=20)]),
            _gtext(1, 6, 182, 280, "body", [
                _p("First point", bullet=True, size=13, space_after=8),
                _p("Second point", bullet=True, size=13, space_after=8),
                _p("Third point", bullet=True, size=13, space_after=8),
            ]),
            {"type": "line", "x1": 480, "y1": 132, "x2": _MR, "y2": 132, "color": "dk2", "w": 1},
            _gtext(7, 12, 142, 34, "headline", [_p("Second column", size=20)]),
            _gtext(7, 12, 182, 280, "body", [
                _p("First point", bullet=True, size=13, space_after=8),
                _p("Second point", bullet=True, size=13, space_after=8),
                _p("Third point", bullet=True, size=13, space_after=8),
            ]),
        ],
    },
    "three_column": {
        "name": "Three columns",
        "description": "A title over three even ruled columns — a heading and a short bulleted list in each; fill the columns with whatever elements suit the point. Use for three parallel topics, options, or pillars.",
        "comment": "Three columns on the grid: cols 1-4 / 5-8 / 9-12, 12pt gutters. The content per column is just a placeholder — replace it with non-text elements as you see fit. If you keep the column headings and rules, give all three the same colour unless a colour difference is meant to draw attention.",
        "elements": [
            _gtext(1, 12, 44, 60, "headline", [_p("Slide title")]),
            {"type": "line", "x1": 48, "y1": 132, "x2": 324, "y2": 132, "color": "dk2", "w": 1},
            _gtext(1, 4, 142, 34, "headline", [_p("First column", size=20)]),
            _gtext(1, 4, 182, 280, "body", [
                _p("First point", bullet=True, size=13, space_after=8),
                _p("Second point", bullet=True, size=13, space_after=8),
                _p("Third point", bullet=True, size=13, space_after=8),
            ]),
            {"type": "line", "x1": 336, "y1": 132, "x2": 612, "y2": 132, "color": "dk2", "w": 1},
            _gtext(5, 8, 142, 34, "headline", [_p("Second column", size=20)]),
            _gtext(5, 8, 182, 280, "body", [
                _p("First point", bullet=True, size=13, space_after=8),
                _p("Second point", bullet=True, size=13, space_after=8),
                _p("Third point", bullet=True, size=13, space_after=8),
            ]),
            {"type": "line", "x1": 624, "y1": 132, "x2": _MR, "y2": 132, "color": "dk2", "w": 1},
            _gtext(9, 12, 142, 34, "headline", [_p("Third column", size=20)]),
            _gtext(9, 12, 182, 280, "body", [
                _p("First point", bullet=True, size=13, space_after=8),
                _p("Second point", bullet=True, size=13, space_after=8),
                _p("Third point", bullet=True, size=13, space_after=8),
            ]),
        ],
    },
    "image_right": {
        "name": "Image right",
        "description": "Bullets on the left, an image or another element (like a chart) on the right. Use when a visual supports a few points.",
        "comment": "The image can sit on either side: swap the x of the text column (48) and the image (480) to flip the layout.",
        "elements": [
            _gtext(1, 12, 44, 60, "headline", [_p("Slide title")]),
            _gtext(1, 6, 130, 340, "body", [
                _p("Describe the visual", bullet=True, space_after=8),
                _p("Second supporting point", bullet=True, space_after=8),
            ]),
            {"type": "image", "x": 480, "y": 130, "w": 432, "h": 300, "token": "", "fit": "contain"},
        ],
    },
    "image_bleed": {
        "name": "Image half-bleed",
        "description": "Title + bullets on the left, an image (or another element, like a chart) bleeding off the right half of the slide (full height, edge to edge). Use for a high-impact photo or hero visual with a short message.",
        "comment": "The image takes the larger side edge-to-edge (fit=cover crops it). To mirror the layout, put the image at x=0 (w=624) and move the text column to columns 9-12 (x=624).",
        "skip_footer": True,
        "elements": [
            {"type": "image", "x": 336, "y": 0, "w": 624, "h": 540, "token": "", "fit": "cover"},
            _gtext(1, 4, 44, 90, "headline", [_p("Slide title")]),
            _gtext(1, 4, 150, 320, "body", [
                _p("Describe the visual", bullet=True, space_after=8),
                _p("Second supporting point", bullet=True, space_after=8),
            ]),
        ],
    },
    "metric": {
        "name": "Metric",
        "description": "One big hero number on a full-height colour panel, with the supporting bullets beside it. Use when a slide is about a single number, like a KPI or metric.",
        "elements": [
            {"type": "shape", "x": 0, "y": 0, "w": 408, "h": 540, "shape": "rect", "fill": "dk2"},  # panel through col 5
            _gtext(1, 4, 44, 60, "headline", [_p("Key results", color="lt1")]),
            _gtext(1, 4, 180, 160, "headline", [_p("42%", size=120, bold=True, color="accent1")], valign="bottom"),
            _gtext(1, 4, 356, 40, "subhead", [_p("What this number means", size=18, color="lt1")]),
            # Bullets start at col 7 so they don't touch the panel's edge (col 6
            # starts at x=408, exactly where the panel ends).
            _gtext(7, 12, 186, 260, "body", [
                _p("Context for the metric", bullet=True, space_after=10),
                _p("Why it matters", bullet=True, space_after=10),
                _p("What we do next", bullet=True, space_after=10),
            ]),
        ],
    },
    "chart": {
        "name": "Chart",
        "description": "A chart on one side with takeaway bullets on the other (the chart can be swapped for another element, like a picture). Use for any trend, comparison, distribution or bridge — the takeaway goes in the bullets.",
        "comment": "Pick the chart kind that fits the question: change the type from 'column' to 'line' (trend), 'bar' (ranked, horizontal), 'scatter' (relationship between two measures — uses point [x,y]/[x,y,size]), 'histogram' (distribution — set 'bins'), 'dot' (a ranked dot plot), 'bullet' (target vs actual — set 'targets'/'bands'), or 'waterfall' (contribution to a change). Set legend:true only with more than one series. Fill in the source line (what data, which period, who analysed it) — or drop it only if the chart truly needs no sourcing. To put the chart on the RIGHT instead: bullets on cols 1-4 (x=48 w=276) and the chart at x=336 w=576. For a full-width chart, widen it to x=48 w=864 and drop the bullets.",
        "elements": [
            _gtext(1, 12, 44, 60, "headline", [_p("Slide title")]),
            {"type": "chart", "x": 48, "y": 130, "w": 564, "h": 320, "chart": "column",
             "title": "", "legend": False,
             "categories": ["Q1", "Q2", "Q3", "Q4"],
             "series": [{"name": "Series 1", "values": [3, 5, 4, 7]}]},
            _gtext(9, 12, 150, 280, "body", [
                _p("What the chart shows", bullet=True, space_after=10),
                _p("The takeaway", bullet=True, space_after=10),
            ]),
            _gtext(1, 12, 488, 20, "caption", [_p("Source: [dataset / report, period]; team analysis.", size=9)]),
        ],
    },
    "photo": {
        "name": "Photo cover",
        "description": "Full-bleed photo cover — a big headline anchored bottom-left over a background image with a dark scrim. Use for a photographic cover or a dramatic section opener.",
        "skip_footer": True,
        "bg": "dk2",
        "bg_image": "",  # set a "[[image:UUID]]" token to fill the slide with a photo
        "bg_scrim": {"color": "dk1", "opacity": 0.45},
        "elements": [
            _gtext(1, 10, 330, 110, "headline", [_p("Headline over a photo", size=52, color="lt1")], valign="bottom"),
            _gtext(1, 9, 456, 40, "body", [_p("One line that sets up the story", size=16, color="lt2")]),
        ],
    },
    "agenda": {
        "name": "Agenda",
        "description": "A numbered agenda / contents list — big editorial numerals, a topic and a one-line gloss per row, with an optional duration on the right. Use when a slide should show an agenda or list the presentation content.",
        "comment": "Four 88pt rows from y=144, separated by hairlines. Drop a row (numeral + topic + gloss + duration, and the hairline above it) for a shorter agenda; drop the right-hand duration column if timings aren't relevant.",
        "elements": [
            _gtext(1, 12, 44, 60, "headline", [_p("Agenda")]),
            _text(_ML, 144, 96, 52, "headline", [_p("01", size=40, color="accent1")]),
            _gtext(3, 10, 142, 34, "headline", [_p("First topic", size=24)]),
            _gtext(3, 10, 182, 28, "body", [_p("One line on what this part covers", size=13)]),
            _gtext(11, 12, 150, 24, "caption", [_p("10 min", size=12, align="right")]),
            {"type": "line", "x1": _ML, "y1": 222, "x2": _MR, "y2": 222, "color": "accent3", "w": 1},
            _text(_ML, 232, 96, 52, "headline", [_p("02", size=40, color="accent1")]),
            _gtext(3, 10, 230, 34, "headline", [_p("Second topic", size=24)]),
            _gtext(3, 10, 270, 28, "body", [_p("One line on what this part covers", size=13)]),
            _gtext(11, 12, 238, 24, "caption", [_p("15 min", size=12, align="right")]),
            {"type": "line", "x1": _ML, "y1": 310, "x2": _MR, "y2": 310, "color": "accent3", "w": 1},
            _text(_ML, 320, 96, 52, "headline", [_p("03", size=40, color="accent1")]),
            _gtext(3, 10, 318, 34, "headline", [_p("Third topic", size=24)]),
            _gtext(3, 10, 358, 28, "body", [_p("One line on what this part covers", size=13)]),
            _gtext(11, 12, 326, 24, "caption", [_p("20 min", size=12, align="right")]),
            {"type": "line", "x1": _ML, "y1": 398, "x2": _MR, "y2": 398, "color": "accent3", "w": 1},
            _text(_ML, 408, 96, 52, "headline", [_p("04", size=40, color="accent1")]),
            _gtext(3, 10, 406, 34, "headline", [_p("Fourth topic", size=24)]),
            _gtext(3, 10, 446, 28, "body", [_p("One line on what this part covers", size=13)]),
            _gtext(11, 12, 414, 24, "caption", [_p("5 min", size=12, align="right")]),
        ],
    },
    "exec_summary": {
        "name": "Executive summary",
        "description": "The bottom line — an action-title recommendation on the left, numbered supporting messages stacked on the right. Use as the first content slide of a recommendation deck — the answer up front.",
        "comment": "For more or fewer messages, re-space the rows to fit between y=88 and y≈460: each message is a `subhead` number (h=30) over a `body` line (h=70), separated by a hairline from x=408 to x=912.",
        "elements": [
            _gtext(1, 4, 88, 260, "headline", [_p("The bottom line — write the recommendation as the title", size=34)]),
            _gtext(6, 12, 88, 30, "subhead", [_p("1")]),
            _gtext(6, 12, 120, 70, "body", [_p("Key message one — lead with the answer, then the support.")]),
            {"type": "line", "x1": 408, "y1": 200, "x2": _MR, "y2": 200, "color": "accent3", "w": 1},
            _gtext(6, 12, 216, 30, "subhead", [_p("2")]),
            _gtext(6, 12, 248, 70, "body", [_p("Key message two — quantify the impact where you can.")]),
            {"type": "line", "x1": 408, "y1": 328, "x2": _MR, "y2": 328, "color": "accent3", "w": 1},
            _gtext(6, 12, 344, 30, "subhead", [_p("3")]),
            _gtext(6, 12, 376, 70, "body", [_p("Key message three — end with the recommended next step.")]),
        ],
    },
    "kpi_row": {
        "name": "KPI row",
        "description": "Two to four KPIs side by side — a big number, its label and one line of context each, split by vertical hairlines. Use when a few headline numbers tell the story together.",
        "comment": "This example has three KPIs on thirds. For two, use the halves (x=48 w=420 and x=480 w=432, one hairline at x=474); for four, the quarters (x=48/264/480/696, w=204/204/204/216, hairlines at x=258/474/690) and drop the number size a step (e.g. 48) so it fits.",
        "elements": [
            _gtext(1, 12, 44, 60, "headline", [_p("Key results")]),
            # Three columns (cols 1-4 / 5-8 / 9-12) with a hairline centred in each gutter.
            _gtext(1, 4, 180, 92, "headline", [_p("42%", size=62, bold=True, color="accent1")]),
            _gtext(1, 4, 286, 28, "body", [_p("Metric one", bold=True, size=16, color="dk2")]),
            _gtext(1, 4, 322, 90, "caption", [_p("One line of context on why this number matters.", size=12)]),
            {"type": "line", "x1": 330, "y1": 180, "x2": 330, "y2": 440, "color": "accent3", "w": 1},
            _gtext(5, 8, 180, 92, "headline", [_p("3.4x", size=62, bold=True, color="accent1")]),
            _gtext(5, 8, 286, 28, "body", [_p("Metric two", bold=True, size=16, color="dk2")]),
            _gtext(5, 8, 322, 90, "caption", [_p("One line of context on why this number matters.", size=12)]),
            {"type": "line", "x1": 618, "y1": 180, "x2": 618, "y2": 440, "color": "accent3", "w": 1},
            _gtext(9, 12, 180, 92, "headline", [_p("$6M", size=62, bold=True, color="accent1")]),
            _gtext(9, 12, 286, 28, "body", [_p("Metric three", bold=True, size=16, color="dk2")]),
            _gtext(9, 12, 322, 90, "caption", [_p("One line of context on why this number matters.", size=12)]),
        ],
    },
    "swimlane": {
        "name": "Swimlane",
        "description": "A swimlane process — role/function lanes down the side, stages across the top, and the steps that hand off between lanes joined by arrows. Use for an operating model or who-does-what across stages.",
        "comment": "Rename the three lane labels to the owners/functions and the four column headers to your stages. Move each step box into the lane×stage cell that owns it, and connect consecutive steps with arrows — a straight arrow within one lane, an elbow when the work hands off to another lane. Drop a lane (its band, label and separators) or a stage column to resize the grid.",
        "elements": [
            _gtext(1, 12, 44, 60, "headline", [_p("Operating model")]),
            # Three pale lane bands, each with a dark label cell at the left; four
            # stage columns split by vertical hairlines. Step boxes sit in the
            # lane×stage cell that owns them and hand off along the arrows.
            {"type": "shape", "x": _ML, "y": 150, "w": _MW, "h": 100, "shape": "rect", "fill": "lt2"},
            {"type": "shape", "x": _ML, "y": 256, "w": _MW, "h": 100, "shape": "rect", "fill": "lt2"},
            {"type": "shape", "x": _ML, "y": 362, "w": _MW, "h": 100, "shape": "rect", "fill": "lt2"},
            {"type": "line", "x1": 345, "y1": 150, "x2": 345, "y2": 462, "color": "accent3", "w": 1},
            {"type": "line", "x1": 534, "y1": 150, "x2": 534, "y2": 462, "color": "accent3", "w": 1},
            {"type": "line", "x1": 723, "y1": 150, "x2": 723, "y2": 462, "color": "accent3", "w": 1},
            {"type": "shape", "x": _ML, "y": 150, "w": 108, "h": 100, "shape": "rect", "fill": "dk2",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Lane 1", "color": "lt1", "b": True, "size": 14}]}]}},
            {"type": "shape", "x": _ML, "y": 256, "w": 108, "h": 100, "shape": "rect", "fill": "dk2",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Lane 2", "color": "lt1", "b": True, "size": 14}]}]}},
            {"type": "shape", "x": _ML, "y": 362, "w": 108, "h": 100, "shape": "rect", "fill": "dk2",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Lane 3", "color": "lt1", "b": True, "size": 14}]}]}},
            _text(156, 118, 189, 26, "body", [_p("Phase 1", size=13, bold=True, color="dk2", align="center")]),
            _text(345, 118, 189, 26, "body", [_p("Phase 2", size=13, bold=True, color="dk2", align="center")]),
            _text(534, 118, 189, 26, "body", [_p("Phase 3", size=13, bold=True, color="dk2", align="center")]),
            _text(723, 118, 189, 26, "body", [_p("Phase 4", size=13, bold=True, color="dk2", align="center")]),
            # Steps are peers and share one colour (their lane already encodes
            # ownership); recolour a single step only to highlight it.
            {"type": "shape", "x": 176, "y": 172, "w": 150, "h": 56, "shape": "rounded_rect", "fill": "accent2",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Step 1", "color": "lt1", "b": True}]}]}},
            {"type": "shape", "x": 365, "y": 278, "w": 150, "h": 56, "shape": "rounded_rect", "fill": "accent2",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Step 2", "color": "lt1", "b": True}]}]}},
            {"type": "shape", "x": 554, "y": 278, "w": 150, "h": 56, "shape": "rounded_rect", "fill": "accent2",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Step 3", "color": "lt1", "b": True}]}]}},
            {"type": "shape", "x": 743, "y": 384, "w": 150, "h": 56, "shape": "rounded_rect", "fill": "accent2",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Step 4", "color": "lt1", "b": True}]}]}},
            # Handoff arrows: step 1 → 2 (down into lane 2), 2 → 3 (along lane 2), 3 → 4 (down into lane 3).
            {"type": "line", "x1": 326, "y1": 200, "x2": 440, "y2": 200, "color": "accent3", "w": 1.5},
            {"type": "line", "x1": 440, "y1": 200, "x2": 440, "y2": 278, "color": "accent3", "w": 1.5, "arrow": "end"},
            {"type": "line", "x1": 515, "y1": 306, "x2": 554, "y2": 306, "color": "accent3", "w": 1.5, "arrow": "end"},
            {"type": "line", "x1": 704, "y1": 306, "x2": 818, "y2": 306, "color": "accent3", "w": 1.5},
            {"type": "line", "x1": 818, "y1": 306, "x2": 818, "y2": 384, "color": "accent3", "w": 1.5, "arrow": "end"},
        ],
    },
    "roadmap_gantt": {
        "name": "Roadmap / Gantt",
        "description": "A multi-workstream roadmap — tracks down the side, quarters across the top, and duration bars spanning the periods each workstream runs, with milestone diamonds and a 'now' marker. Use for a schedule or roadmap with several parallel workstreams (for a single track use the timeline recipe in the skill).",
        "comment": "Rename the four workstream labels and the Q1–Q4 headers, then stretch each duration bar to span its real periods: a bar's left edge is its start quarter's x (156 / 345 / 534 / 723) and its right edge the end quarter's right edge (345 / 534 / 723 / 912). Move the diamonds to real milestone dates, slide the dashed 'Now' line to today, and drop a row (label + bar) for fewer workstreams.",
        "elements": [
            _gtext(1, 12, 44, 60, "headline", [_p("Roadmap")]),
            # Quarter headers over four columns (156–345 / 345–534 / 534–723 /
            # 723–912); a rule under them and dashed quarter gridlines behind the bars.
            _text(156, 120, 189, 24, "body", [_p("Q1", size=13, bold=True, color="dk2", align="center")]),
            _text(345, 120, 189, 24, "body", [_p("Q2", size=13, bold=True, color="dk2", align="center")]),
            _text(534, 120, 189, 24, "body", [_p("Q3", size=13, bold=True, color="dk2", align="center")]),
            _text(723, 120, 189, 24, "body", [_p("Q4", size=13, bold=True, color="dk2", align="center")]),
            {"type": "line", "x1": _ML, "y1": 150, "x2": _MR, "y2": 150, "color": "accent3", "w": 1},
            {"type": "line", "x1": 345, "y1": 150, "x2": 345, "y2": 462, "color": "accent3", "w": 1, "dash": "dash"},
            {"type": "line", "x1": 534, "y1": 150, "x2": 534, "y2": 462, "color": "accent3", "w": 1, "dash": "dash"},
            {"type": "line", "x1": 723, "y1": 150, "x2": 723, "y2": 462, "color": "accent3", "w": 1, "dash": "dash"},
            # Workstream labels (left) + duration bars spanning their quarters.
            # Bars are peers and share one colour (the row label already names the
            # workstream); accent1 is reserved for the "Now" marker below.
            _text(_ML, 183, 100, 30, "body", [_p("Workstream 1", size=12, bold=True)], valign="middle"),
            {"type": "shape", "x": 156, "y": 183, "w": 378, "h": 30, "shape": "rounded_rect", "fill": "accent2",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Phase 1", "color": "lt1", "b": True, "size": 12}]}]}},
            _text(_ML, 259, 100, 30, "body", [_p("Workstream 2", size=12, bold=True)], valign="middle"),
            {"type": "shape", "x": 345, "y": 259, "w": 378, "h": 30, "shape": "rounded_rect", "fill": "accent2",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Build", "color": "lt1", "b": True, "size": 12}]}]}},
            _text(_ML, 335, 100, 30, "body", [_p("Workstream 3", size=12, bold=True)], valign="middle"),
            {"type": "shape", "x": 156, "y": 335, "w": 756, "h": 30, "shape": "rounded_rect", "fill": "accent2",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Ongoing engagement", "color": "lt1", "b": True, "size": 12}]}]}},
            _text(_ML, 411, 100, 30, "body", [_p("Workstream 4", size=12, bold=True)], valign="middle"),
            {"type": "shape", "x": 534, "y": 411, "w": 378, "h": 30, "shape": "rounded_rect", "fill": "accent2",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Launch", "color": "lt1", "b": True, "size": 12}]}]}},
            # Milestone diamonds sit on a bar at a key date.
            {"type": "shape", "x": 525, "y": 189, "w": 18, "h": 18, "shape": "diamond", "fill": "dk2"},
            {"type": "shape", "x": 714, "y": 341, "w": 18, "h": 18, "shape": "diamond", "fill": "dk2"},
            # "Now" marker (dashed), drawn last so it reads over the bars.
            _text(560, 128, 80, 20, "caption", [_p("Now", size=11, bold=True, color="accent1", align="center")]),
            {"type": "line", "x1": 600, "y1": 150, "x2": 600, "y2": 470, "color": "accent1", "w": 1.5, "dash": "dash"},
        ],
    },
    "issue_tree": {
        "name": "Issue tree",
        "description": "A left-to-right issue tree (MECE decomposition) — a key question breaking into branches and their sub-drivers, joined by elbow connectors. Use for a hypothesis / driver tree or any 'break the question down' slide.",
        "comment": "A MECE issue tree: the key question on the left, three mutually-exclusive branches, each split into two sub-drivers/tests. Keep the branches distinct and collectively exhaustive, and phrase each leaf as a testable driver. Drop a leaf pair (with its stub + bus + feeder lines) for a one-level tree, or copy a branch box and its connectors to add a fourth.",
        "elements": [
            _gtext(1, 12, 44, 60, "headline", [_p("Issue tree")]),
            # Root question (left) → 3 branches (middle) → 2 leaves each (right),
            # wired with elbow connectors built from straight segments: a stub off
            # the parent, a vertical bus, then an arrowed feeder into each child.
            {"type": "shape", "x": 48, "y": 252, "w": 204, "h": 96, "shape": "rounded_rect", "fill": "accent1",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "The key question", "color": "lt1", "b": True, "size": 16}]}]}},
            {"type": "shape", "x": 336, "y": 136, "w": 204, "h": 72, "shape": "rounded_rect", "fill": "dk2",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Branch 1", "color": "lt1", "b": True, "size": 15}]}]}},
            {"type": "shape", "x": 336, "y": 264, "w": 204, "h": 72, "shape": "rounded_rect", "fill": "dk2",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Branch 2", "color": "lt1", "b": True, "size": 15}]}]}},
            {"type": "shape", "x": 336, "y": 392, "w": 204, "h": 72, "shape": "rounded_rect", "fill": "dk2",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Branch 3", "color": "lt1", "b": True, "size": 15}]}]}},
            {"type": "shape", "x": 624, "y": 121, "w": 288, "h": 44, "shape": "rounded_rect", "fill": "lt2",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Driver or test", "color": "dk2", "size": 12}]}]}},
            {"type": "shape", "x": 624, "y": 179, "w": 288, "h": 44, "shape": "rounded_rect", "fill": "lt2",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Driver or test", "color": "dk2", "size": 12}]}]}},
            {"type": "shape", "x": 624, "y": 249, "w": 288, "h": 44, "shape": "rounded_rect", "fill": "lt2",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Driver or test", "color": "dk2", "size": 12}]}]}},
            {"type": "shape", "x": 624, "y": 307, "w": 288, "h": 44, "shape": "rounded_rect", "fill": "lt2",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Driver or test", "color": "dk2", "size": 12}]}]}},
            {"type": "shape", "x": 624, "y": 377, "w": 288, "h": 44, "shape": "rounded_rect", "fill": "lt2",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Driver or test", "color": "dk2", "size": 12}]}]}},
            {"type": "shape", "x": 624, "y": 435, "w": 288, "h": 44, "shape": "rounded_rect", "fill": "lt2",
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Driver or test", "color": "dk2", "size": 12}]}]}},
            # Root → branches: stub, vertical bus across the branch centres, arrowed feeders.
            {"type": "line", "x1": 252, "y1": 300, "x2": 294, "y2": 300, "color": "accent3", "w": 1.5},
            {"type": "line", "x1": 294, "y1": 172, "x2": 294, "y2": 428, "color": "accent3", "w": 1.5},
            {"type": "line", "x1": 294, "y1": 172, "x2": 336, "y2": 172, "color": "accent3", "w": 1.5, "arrow": "end"},
            {"type": "line", "x1": 294, "y1": 300, "x2": 336, "y2": 300, "color": "accent3", "w": 1.5, "arrow": "end"},
            {"type": "line", "x1": 294, "y1": 428, "x2": 336, "y2": 428, "color": "accent3", "w": 1.5, "arrow": "end"},
            # Branch 1 → its two leaves.
            {"type": "line", "x1": 540, "y1": 172, "x2": 582, "y2": 172, "color": "accent3", "w": 1.5},
            {"type": "line", "x1": 582, "y1": 143, "x2": 582, "y2": 201, "color": "accent3", "w": 1.5},
            {"type": "line", "x1": 582, "y1": 143, "x2": 624, "y2": 143, "color": "accent3", "w": 1.5, "arrow": "end"},
            {"type": "line", "x1": 582, "y1": 201, "x2": 624, "y2": 201, "color": "accent3", "w": 1.5, "arrow": "end"},
            # Branch 2 → its two leaves.
            {"type": "line", "x1": 540, "y1": 300, "x2": 582, "y2": 300, "color": "accent3", "w": 1.5},
            {"type": "line", "x1": 582, "y1": 271, "x2": 582, "y2": 329, "color": "accent3", "w": 1.5},
            {"type": "line", "x1": 582, "y1": 271, "x2": 624, "y2": 271, "color": "accent3", "w": 1.5, "arrow": "end"},
            {"type": "line", "x1": 582, "y1": 329, "x2": 624, "y2": 329, "color": "accent3", "w": 1.5, "arrow": "end"},
            # Branch 3 → its two leaves.
            {"type": "line", "x1": 540, "y1": 428, "x2": 582, "y2": 428, "color": "accent3", "w": 1.5},
            {"type": "line", "x1": 582, "y1": 399, "x2": 582, "y2": 457, "color": "accent3", "w": 1.5},
            {"type": "line", "x1": 582, "y1": 399, "x2": 624, "y2": 399, "color": "accent3", "w": 1.5, "arrow": "end"},
            {"type": "line", "x1": 582, "y1": 457, "x2": 624, "y2": 457, "color": "accent3", "w": 1.5, "arrow": "end"},
        ],
    },
    "quote": {
        "name": "Quote",
        "description": "A large pull-quote with attribution. Use for a customer / expert quote or a memorable line that anchors a section.",
        "bg": "lt2",
        "elements": [
            _text(96, 70, 200, 160, "headline", [_p("“", size=150, bold=True, color="accent3", align="left")]),
            _gtext(2, 11, 200, 170, "quote", [_p("A short, memorable quotation that anchors the slide.", align="center")], valign="middle"),
            _gtext(2, 11, 384, 30, "caption", [_p("— Attribution", align="center")]),
        ],
    },
    "closing": {
        "name": "Closing",
        "description": "A closing / thank-you slide with contact details on a dark background. Use as the last slide.",
        "skip_footer": True,
        "bg": "dk2",
        "elements": [
            _gtext(1, 12, 220, 80, "headline", [_p("Thank you", color="lt1", align="center")]),
            _gtext(1, 12, 315, 40, "subhead", [_p("name@example.com", color="lt2", align="center")]),
        ],
    },
    "blank": {
        "name": "Blank",
        "description": "An empty slide to build from scratch. Use only when no premade layout fits — and even then build it from the premade elements and the skill's diagram recipes.",
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
