"""Deck JSON schema: validation, canonical serialization, id minting, hashing.

The deck is one JSON document (stored in ``SlideSet.content``). This module is
the single source of truth for its shape and for the **canonical text** the LLM
edits against — see :func:`canonical_deck_text`.

Why a canonical serializer: ``SlideSet.content`` is a Postgres ``jsonb`` column,
which discards whitespace and **reorders object keys**. If the model anchored a
find/replace edit against the raw stored JSON it wrote, a jsonb round-trip would
no longer match. So every code path that shows the model deck text — the
prompt injection and the ``slide_canvas_edit`` tool — goes through
:func:`canonical_deck_text` (``sort_keys=True``), which is deterministic
regardless of jsonb's internal key order. Array order (slides, elements, runs)
IS preserved by jsonb, so element z-order and slide order are stable.

Pure Python (pydantic only) — unit-testable anywhere.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chat.slides.theme import resolve_theme

# --- Limits -----------------------------------------------------------------
DECK_MAX_CHARS = 120_000
MAX_SLIDE_SETS_PER_THREAD = 5
MAX_ACTIVE_SLIDE_SETS = 1
MAX_SLIDES_PER_DECK = 30
MAX_ELEMENTS_PER_SLIDE = 25
MAX_PREVIEW_SLIDES_PER_CALL = 4
MAX_CHART_SERIES = 8
MAX_CHART_POINTS = 30

# Slide/element ids: short, url-safe, stable. Comments and previews key on them.
_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,15}$")

_DASH = ("solid", "dash", "dot", "dashdot")
_ALIGN = ("left", "center", "right", "justify")
_VALIGN = ("top", "middle", "bottom")


# --- Canonical serialization ------------------------------------------------
def canonical_deck_text(obj) -> str:
    """Deterministic pretty JSON — the text the model sees and edits against."""
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)


# --- Pydantic models (validation only; we persist the raw dict) -------------
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Run(_Strict):
    t: str = ""
    b: bool | None = None
    i: bool | None = None
    u: bool | None = None
    size: float | None = None
    color: str | None = None
    font: str | None = None


class Paragraph(_Strict):
    runs: list[Run] = Field(default_factory=list)
    align: Literal[_ALIGN] = "left"  # type: ignore[valid-type]
    bullet: bool = False
    level: int = Field(default=0, ge=0, le=4)
    space_after: float | None = None


class TextBody(_Strict):
    cls: str | None = Field(default=None, alias="class")
    paragraphs: list[Paragraph] = Field(default_factory=list)
    valign: Literal[_VALIGN] = "top"  # type: ignore[valid-type]
    wrap: bool = True


class LineStyle(_Strict):
    color: str | None = None
    w: float | None = None
    dash: Literal[_DASH] | None = None  # type: ignore[valid-type]


class TextElement(_Strict):
    id: str | None = None
    type: Literal["text"]
    x: float
    y: float
    w: float
    h: float
    cls: str | None = Field(default=None, alias="class")
    paragraphs: list[Paragraph] = Field(default_factory=list)
    valign: Literal[_VALIGN] = "top"  # type: ignore[valid-type]
    wrap: bool = True
    rotation: float | None = None


class ShapeElement(_Strict):
    id: str | None = None
    type: Literal["shape"]
    x: float
    y: float
    w: float
    h: float
    shape: str = "rect"
    fill: str | None = None
    line: LineStyle | None = None
    box: str | None = None
    text: TextBody | None = None
    rotation: float | None = None
    opacity: float | None = Field(default=None, ge=0.0, le=1.0)  # fill translucency


class ImageElement(_Strict):
    id: str | None = None
    type: Literal["image"]
    x: float
    y: float
    w: float
    h: float
    token: str = ""
    fit: Literal["contain", "cover", "stretch"] = "contain"
    rotation: float | None = None
    opacity: float | None = Field(default=None, ge=0.0, le=1.0)  # fade a placed image


class Cell(_Strict):
    t: str = ""
    cls: str | None = Field(default=None, alias="class")
    b: bool | None = None
    i: bool | None = None
    size: float | None = None
    color: str | None = None
    fill: str | None = None
    align: Literal["left", "center", "right"] | None = None


class TableElement(_Strict):
    id: str | None = None
    type: Literal["table"]
    x: float
    y: float
    w: float
    h: float
    col_widths: list[float] | None = None
    header: bool = True
    banding: bool = True
    style: str = "default"
    rows: list[list[Cell]] = Field(default_factory=list)


class LineElement(_Strict):
    id: str | None = None
    type: Literal["line"]
    x1: float
    y1: float
    x2: float
    y2: float
    color: str | None = None
    w: float | None = None
    dash: Literal[_DASH] | None = None  # type: ignore[valid-type]
    arrow: Literal["none", "end", "start", "both"] = "none"


_CHART_KINDS = ("column", "bar", "line", "area", "pie", "waterfall")


class ChartSeries(_Strict):
    name: str = ""
    values: list[float] = Field(default_factory=list)


class ChartElement(_Strict):
    id: str | None = None
    type: Literal["chart"]
    x: float
    y: float
    w: float
    h: float
    chart: Literal[_CHART_KINDS] = "column"  # type: ignore[valid-type]
    categories: list[str] = Field(default_factory=list)
    series: list[ChartSeries] = Field(default_factory=list)
    title: str = ""
    legend: bool = True
    value_labels: bool = False
    # Stack the series into one bar per category (a composition of a total)
    # instead of grouping them side by side. Only meaningful for column/bar/area.
    stacked: bool = False
    # For chart="waterfall" (a bridge): category indices drawn as absolute bars
    # from zero (a base/subtotal that resets the running total); every other
    # category is a delta that floats on the running total (green up / red down).
    # Empty -> the first category is treated as the base.
    totals: list[int] = Field(default_factory=list)
    colors: list[str] | None = None  # override theme.colors.chart_ramp


Element = Annotated[
    Union[TextElement, ShapeElement, ImageElement, TableElement, LineElement, ChartElement],
    Field(discriminator="type"),
]


class Scrim(_Strict):
    """A translucent colour wash over a background image, for legible text."""
    color: str = "dk1"
    opacity: float = Field(default=0.4, ge=0.0, le=1.0)


class Slide(_Strict):
    id: str | None = None
    name: str = ""
    bg: str | None = None
    bg_image: str = ""          # image token drawn full-bleed behind everything
    bg_scrim: Scrim | None = None
    skip_footer: bool = False
    notes: str = ""
    elements: list[Element] = Field(default_factory=list)


class Size(_Strict):
    w: float = 960
    h: float = 540


class Deck(_Strict):
    version: int = 1
    size: Size = Field(default_factory=Size)
    theme: dict = Field(default_factory=dict)
    slides: list[Slide] = Field(default_factory=list)


# --- Validation -------------------------------------------------------------
def validate_deck(deck: dict) -> list[dict]:
    """Validate a deck dict. Returns a list of ``{"path","message"}`` issues.

    Empty list == valid. Structural errors come from pydantic (with a
    dotted/indexed path the model can act on); a second pass adds semantic
    checks pydantic can't express (id format/uniqueness, limits, size).
    """
    if not isinstance(deck, dict):
        return [{"path": "", "message": "Deck must be a JSON object."}]

    issues: list[dict] = []
    try:
        Deck.model_validate(deck)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"] if p != "__root__")
            issues.append({"path": loc or "(root)", "message": err["msg"]})
        return issues  # structural errors first — don't pile semantic noise on top

    issues.extend(_semantic_issues(deck))
    return issues


def _semantic_issues(deck: dict) -> list[dict]:
    issues: list[dict] = []
    slides = deck.get("slides") or []

    if len(slides) > MAX_SLIDES_PER_DECK:
        issues.append({"path": "slides", "message": f"Too many slides (max {MAX_SLIDES_PER_DECK})."})

    seen_slide_ids: set[str] = set()
    for si, slide in enumerate(slides):
        sid = slide.get("id")
        if sid is not None:
            if not _ID_RE.match(sid):
                issues.append({"path": f"slides.{si}.id", "message": "Invalid id (use a-z, 0-9, _-, ≤16 chars, letter first)."})
            elif sid in seen_slide_ids:
                issues.append({"path": f"slides.{si}.id", "message": f"Duplicate slide id '{sid}'."})
            seen_slide_ids.add(sid)

        elements = slide.get("elements") or []
        if len(elements) > MAX_ELEMENTS_PER_SLIDE:
            issues.append({"path": f"slides.{si}.elements", "message": f"Too many elements (max {MAX_ELEMENTS_PER_SLIDE})."})

        seen_el_ids: set[str] = set()
        for ei, el in enumerate(elements):
            eid = el.get("id")
            if eid is not None:
                if not _ID_RE.match(eid):
                    issues.append({"path": f"slides.{si}.elements.{ei}.id", "message": "Invalid id."})
                elif eid in seen_el_ids:
                    issues.append({"path": f"slides.{si}.elements.{ei}.id", "message": f"Duplicate element id '{eid}'."})
                seen_el_ids.add(eid)
            if el.get("type") == "table":
                for cell in _iter_cells(el):
                    if "span" in cell:  # reserved for future merge support
                        issues.append({"path": f"slides.{si}.elements.{ei}", "message": "Cell merges ('span') are not supported yet."})
                        break
            if el.get("type") == "chart":
                sers = el.get("series") or []
                path = f"slides.{si}.elements.{ei}"
                if len(sers) > MAX_CHART_SERIES:
                    issues.append({"path": path, "message": f"Too many chart series (max {MAX_CHART_SERIES})."})
                if not sers:
                    issues.append({"path": path, "message": "A chart needs at least one series."})
                for s in sers:
                    if len(s.get("values") or []) > MAX_CHART_POINTS:
                        issues.append({"path": path, "message": f"Too many chart data points (max {MAX_CHART_POINTS})."})
                        break

    if len(canonical_deck_text(deck)) > DECK_MAX_CHARS:
        issues.append({"path": "(root)", "message": f"Deck is too large (max {DECK_MAX_CHARS} chars)."})

    return issues


def _iter_cells(table: dict):
    for row in table.get("rows") or []:
        for cell in row or []:
            if isinstance(cell, dict):
                yield cell


# --- Id minting -------------------------------------------------------------
def mint_ids(deck: dict) -> dict:
    """Fill in any missing slide/element ids in place (collision-safe).

    Existing ids are preserved (they are referenced by comments/previews and
    must never change). Returns the same dict for convenience.
    """
    slides = deck.get("slides") or []
    used_slide_ids = {s["id"] for s in slides if s.get("id")}
    next_slide = _counter("s", used_slide_ids)
    for slide in slides:
        if not slide.get("id"):
            slide["id"] = next_slide()
            used_slide_ids.add(slide["id"])
        elements = slide.get("elements") or []
        used_el_ids = {e["id"] for e in elements if e.get("id")}
        next_el = _counter("e", used_el_ids)
        for el in elements:
            if not el.get("id"):
                el["id"] = next_el()
                used_el_ids.add(el["id"])
    return deck


def _counter(prefix: str, used: set[str]):
    state = {"n": 0}

    def _next() -> str:
        while True:
            state["n"] += 1
            candidate = f"{prefix}{state['n']}"
            if candidate not in used:
                return candidate

    return _next


# --- Content hashing (render cache + change detection) ----------------------
def slide_content_hash(deck: dict, index: int) -> str:
    """Stable hash of everything that affects slide ``index``'s rendered pixels.

    Includes the resolved theme, the deck size, and the slide's position/total
    (the footer page-number stamp bakes position into the image), so a slide
    that merely shifts position re-renders.
    """
    slides = deck.get("slides") or []
    payload = {
        "slide": slides[index],
        "theme": canonical_deck_text(resolve_theme(deck)),
        "size": deck.get("size"),
        "index": index,
        "total": len(slides),
    }
    return hashlib.sha256(canonical_deck_text(payload).encode("utf-8")).hexdigest()


def slide_hashes(deck: dict) -> dict[str, str]:
    """Map ``slide_id -> content_hash`` (deck must have minted ids)."""
    slides = deck.get("slides") or []
    return {slides[i].get("id", f"_{i}"): slide_content_hash(deck, i) for i in range(len(slides))}


def changed_slide_ids(old: dict | None, new: dict) -> list[str]:
    """Slide ids whose render-affecting content changed (or are new)."""
    old_h = slide_hashes(old) if old else {}
    new_h = slide_hashes(new)
    return [sid for sid, h in new_h.items() if old_h.get(sid) != h]
