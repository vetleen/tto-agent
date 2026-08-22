"""Theme data + CSS-like cascade resolution for slide decks.

A deck's colours, fonts and text styles come from a **base theme**
(:data:`WILFRED_BASE_THEME`) merged with the deck's own *sparse* ``theme``
overrides, then resolved per element via a cascade:

    theme text-style class  ->  element props  ->  paragraph props  ->  run props

with the nearest layer winning. This module is **pure Python** — no Django, no
python-pptx — so it unit-tests anywhere. Colour *names* and font *roles* are
turned into concrete pptx artefacts by the builder (:mod:`chat.slides.pptx_build`);
here we only own the data and the merge/resolve helpers.

Colour values anywhere in a deck are either a **theme colour name** (one of the
12 PowerPoint-native slots ``dk1/lt1/dk2/lt2/accent1..6/hlink/folHlink`` or a
semantic role ``success/warning/danger``) or a ``#RRGGBB`` hex literal. The 12
native slots resolve to *theme* colours (so they track the template's colour
scheme and appear in PowerPoint's picker); everything else resolves to an
explicit RGB.

Font values (in ``text_styles`` and per-run ``font``) are either a **role key**
(``headline/subhead/body/data``) resolved through ``theme["fonts"]`` or a literal
family name.
"""

from __future__ import annotations

import copy

# The 12 PowerPoint-native theme colour slots. These map to MSO_THEME_COLOR in
# the builder and therefore track the template's colour scheme + show in the
# PowerPoint colour picker. Names outside this set (semantic roles, hex) resolve
# to explicit RGB instead.
NATIVE_SLOTS = frozenset(
    {
        "dk1", "lt1", "dk2", "lt2",
        "accent1", "accent2", "accent3", "accent4", "accent5", "accent6",
        "hlink", "folHlink",
    }
)

FONT_ROLES = ("headline", "subhead", "body", "data")

# ---------------------------------------------------------------------------
# The base theme: Wilfred's "forest nocturne" brand as a LIGHT presentation
# palette (light background, deep-forest text, copper + emerald accents).
#
# Fonts use metric-compatible families (Caladea ~ Cambria serif, Carlito ~
# Calibri sans) — the committed .ttf faces under core/assets/fonts, which the
# Pillow preview renderer loads directly. Being Cambria/Calibri-metric keeps the
# preview close to how PowerPoint lays out the downloaded .pptx even if the
# viewer substitutes those faces. Swapping to the real brand faces = drop their
# .ttf in core/assets/fonts + update these four names.
# ---------------------------------------------------------------------------
WILFRED_BASE_THEME: dict = {
    "fonts": {
        "headline": "Caladea",
        "subhead": "Caladea",
        "body": "Carlito",
        "data": "Carlito",
    },
    "colors": {
        # 12 PowerPoint-native slots
        "dk1": "#12241B",       # deep forest — primary text
        "lt1": "#FBFAF6",       # warm off-white — slide background
        "dk2": "#1F3D30",       # forest green — headings/secondary dark
        "lt2": "#ECEFE9",       # pale sage — bands / soft fills
        "accent1": "#B87333",   # copper — primary accent
        "accent2": "#2E6B52",   # emerald
        "accent3": "#7FA891",   # sage
        "accent4": "#1C4A42",   # deep teal
        "accent5": "#D9A441",   # gold
        "accent6": "#A9552F",   # terracotta
        "hlink": "#2E6B52",
        "folHlink": "#7FA891",
        # semantic roles (resolve to explicit RGB)
        "success": "#3E7D5A",
        "warning": "#D9A441",
        "danger": "#B23A2E",
        # ordered ramp for future chart support (names or hex)
        "chart_ramp": ["accent1", "accent2", "accent4", "accent5", "accent3", "accent6"],
    },
    "text_styles": {
        "headline": {"font": "headline", "size": 34, "bold": True, "italic": False, "color": "dk2"},
        "subhead": {"font": "subhead", "size": 20, "bold": False, "italic": False, "color": "accent1"},
        "body": {"font": "body", "size": 14, "bold": False, "italic": False, "color": "dk1"},
        "data": {"font": "data", "size": 12, "bold": False, "italic": False, "color": "dk1"},
        "quote": {"font": "headline", "size": 20, "bold": False, "italic": True, "color": "dk2"},
        "caption": {"font": "data", "size": 10, "bold": False, "italic": False, "color": "accent3"},
    },
    "bullet": {"char": "–"},  # en dash
    "boxes": {
        "callout": {"shape": "rounded_rect", "fill": "accent1", "text_color": "lt1", "class": "body"},
        "panel": {"shape": "rect", "fill": "lt2", "text_color": "dk1", "class": "body"},
        "pill": {"shape": "rounded_rect", "fill": "dk2", "text_color": "lt1", "class": "data"},
        "arrow_r": {"shape": "right_arrow", "fill": "accent2", "text_color": "lt1", "class": "data"},
        "arrow_l": {"shape": "left_arrow", "fill": "accent2", "text_color": "lt1", "class": "data"},
    },
    "table": {
        "header_fill": "dk2",
        "header_color": "lt1",
        "band_fill": "lt2",
        "grid_color": "accent3",
        "text_class": "data",
    },
    # Footer + page number are stamped by the builder on every non-skip_footer
    # slide; they never appear in slide JSON. logo_asset is a filename under
    # chat/slides/assets/ (empty = no logo).
    "footer": {
        "logo_asset": "",
        "text": "",
        "x": 28, "y": 512, "w": 200, "h": 18,
        "size": 9, "color": "dk2", "align": "left",
    },
    "page_number": {
        "x": 900, "y": 512, "w": 36, "h": 18,
        "font": "data", "size": 9, "color": "dk2", "align": "right",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` onto a copy of ``base``.

    Nested dicts merge key-by-key; lists and scalars in ``override`` replace the
    corresponding value wholesale (so a deck can swap the whole ``chart_ramp``).
    """
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def resolve_theme(deck: dict) -> dict:
    """Return the effective theme for ``deck`` (base merged with deck overrides)."""
    return _deep_merge(WILFRED_BASE_THEME, (deck or {}).get("theme") or {})


def font_family(theme: dict, font_ref: str | None) -> str:
    """Resolve a font role key (headline/subhead/body/data) or literal name."""
    if not font_ref:
        return theme["fonts"]["body"]
    return theme["fonts"].get(font_ref, font_ref)


def base_text_style(theme: dict, cls: str | None) -> dict:
    """The theme defaults for a text class (falls back to ``body``)."""
    styles = theme["text_styles"]
    return dict(styles.get(cls or "body", styles["body"]))


def merge_run(base_style: dict, run: dict) -> dict:
    """Apply a run's short-key overrides onto a resolved base text style.

    ``run`` keys: ``font`` (role/literal), ``size``, ``b`` (bold), ``i``
    (italic), ``u`` (underline), ``color``. Absent keys inherit from base.
    """
    out = dict(base_style)
    if run.get("font"):
        out["font"] = run["font"]
    if run.get("size"):
        out["size"] = run["size"]
    if "b" in run and run["b"] is not None:
        out["bold"] = bool(run["b"])
    if "i" in run and run["i"] is not None:
        out["italic"] = bool(run["i"])
    if "u" in run and run["u"] is not None:
        out["underline"] = bool(run["u"])
    if run.get("color"):
        out["color"] = run["color"]
    return out


def resolve_color(theme: dict, value: str | None) -> tuple[str, str] | None:
    """Resolve a colour value to ``("theme", slot)`` or ``("rgb", "RRGGBB")``.

    * ``None``/empty -> ``None`` (no colour set).
    * ``#RRGGBB`` -> ``("rgb", "RRGGBB")``.
    * a native slot name -> ``("theme", slot)`` (the builder maps to
      MSO_THEME_COLOR so it tracks the template scheme + shows in the picker).
    * a semantic/other name present in ``theme["colors"]`` -> ``("rgb", hex)``.
    * anything unresolvable -> ``None``.
    """
    if not value:
        return None
    if value.startswith("#"):
        return ("rgb", value.lstrip("#").upper())
    if value in NATIVE_SLOTS:
        return ("theme", value)
    hexv = theme["colors"].get(value)
    if isinstance(hexv, str) and hexv.startswith("#"):
        return ("rgb", hexv.lstrip("#").upper())
    return None
