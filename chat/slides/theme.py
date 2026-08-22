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


# ---------------------------------------------------------------------------
# Preset themes — colour palettes a user can pick from the slide panel. Each is
# a sparse override of WILFRED_BASE_THEME (colours only), so the structure/fonts/
# text styles stay consistent. "forest" is the base (empty override).
# ---------------------------------------------------------------------------
PRESET_THEMES: dict[str, dict] = {
    "forest": {"label": "Forest", "theme": {}},
    "slate": {"label": "Slate", "theme": {"colors": {
        "dk1": "#1E293B", "lt1": "#F8FAFC", "dk2": "#334155", "lt2": "#E7ECF2",
        "accent1": "#2563EB", "accent2": "#0EA5E9", "accent3": "#94A3B8",
        "accent4": "#1E3A8A", "accent5": "#F59E0B", "accent6": "#64748B",
        "success": "#16A34A", "warning": "#F59E0B", "danger": "#DC2626",
        "hlink": "#2563EB", "folHlink": "#0EA5E9",
        "chart_ramp": ["accent1", "accent2", "accent4", "accent5", "accent3", "accent6"],
    }}},
    "warm": {"label": "Warm sand", "theme": {"colors": {
        "dk1": "#3F2E24", "lt1": "#FBF7F0", "dk2": "#5C4433", "lt2": "#F0E6D8",
        "accent1": "#C2410C", "accent2": "#B45309", "accent3": "#D6BFA8",
        "accent4": "#78350F", "accent5": "#EAB308", "accent6": "#9A3412",
        "success": "#4D7C4E", "warning": "#EAB308", "danger": "#B91C1C",
        "hlink": "#C2410C", "folHlink": "#B45309",
        "chart_ramp": ["accent1", "accent2", "accent4", "accent5", "accent3", "accent6"],
    }}},
    "mono": {"label": "Monochrome", "theme": {"colors": {
        "dk1": "#1A1A1A", "lt1": "#FAFAFA", "dk2": "#2E2E2E", "lt2": "#ECECEC",
        "accent1": "#0D9488", "accent2": "#525252", "accent3": "#A3A3A3",
        "accent4": "#404040", "accent5": "#0F766E", "accent6": "#171717",
        "success": "#15803D", "warning": "#CA8A04", "danger": "#B91C1C",
        "hlink": "#0D9488", "folHlink": "#525252",
        "chart_ramp": ["accent1", "accent2", "accent4", "accent5", "accent3", "accent6"],
    }}},
    "ocean": {"label": "Ocean", "theme": {"colors": {
        "dk1": "#0F2A38", "lt1": "#F4FAFC", "dk2": "#164E5B", "lt2": "#DDEEF0",
        "accent1": "#0E7490", "accent2": "#0891B2", "accent3": "#7DD3D8",
        "accent4": "#155E63", "accent5": "#F0A81E", "accent6": "#2C6E7F",
        "success": "#0F766E", "warning": "#F0A81E", "danger": "#C2410C",
        "hlink": "#0E7490", "folHlink": "#0891B2",
        "chart_ramp": ["accent1", "accent2", "accent4", "accent5", "accent3", "accent6"],
    }}},
}


def preset_swatches() -> list[dict]:
    """``[{name, label, bg, text, accent}]`` for the theme-picker UI (resolved hex)."""
    out = []
    for name, spec in PRESET_THEMES.items():
        colors = _deep_merge(WILFRED_BASE_THEME, spec["theme"])["colors"]
        out.append({
            "name": name, "label": spec["label"],
            "bg": colors["lt1"], "dark": colors["dk2"], "accent": colors["accent1"],
        })
    return out


def preset_theme_override(name: str) -> dict | None:
    """The sparse ``theme`` override dict for a preset name (None if unknown)."""
    spec = PRESET_THEMES.get(name)
    return copy.deepcopy(spec["theme"]) if spec is not None else None


# ---------------------------------------------------------------------------
# Organization default slide theme — the slide sibling of the doc "styles"
# setting (core.styles). An org admin picks one preset (Org settings → Slide
# styles); it seeds the ``theme`` of every new deck authored in that org, so
# decks match the org's brand without the model having to choose. Stored on
# ``Organization.preferences["slide_theme"] = {"name": <preset>}`` (JSON prefs,
# no model field). "forest" (the base) is the default.
# ---------------------------------------------------------------------------
DEFAULT_SLIDE_THEME = "forest"


def get_org_slide_style(org) -> dict:
    """Resolve an org's slide style: ``{"name": <valid preset>}`` (default forest).

    ``org`` may be ``None`` (no membership). Tolerant of malformed stored values —
    an unknown/absent preset name resolves back to :data:`DEFAULT_SLIDE_THEME`.
    """
    stored = (getattr(org, "preferences", None) or {}).get("slide_theme") if org is not None else None
    name = stored.get("name") if isinstance(stored, dict) else None
    if name not in PRESET_THEMES:
        name = DEFAULT_SLIDE_THEME
    return {"name": name}


def validate_org_slide_style(data) -> tuple[dict | None, str | None]:
    """Validate a slide-style payload from the org settings endpoint.

    Returns ``(clean_dict, None)`` on success or ``(None, error_message)``. The
    only lever today is the preset ``name``; kept as a dict so custom colours/
    fonts can be added later without changing the stored shape.
    """
    if not isinstance(data, dict):
        return None, "Invalid slide style payload."
    name = data.get("name", DEFAULT_SLIDE_THEME)
    if not isinstance(name, str) or name not in PRESET_THEMES:
        return None, "Choose one of the available slide themes."
    return {"name": name}, None


def org_slide_theme_override(org) -> dict:
    """The sparse deck ``theme`` override for an org's default slide theme.

    Empty dict for the base ("forest") theme or when there's no org — callers seed
    a new deck's ``theme`` with this only when it's non-empty.
    """
    return preset_theme_override(get_org_slide_style(org)["name"]) or {}


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
