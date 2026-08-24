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
# Organization slide style — the slide sibling of the doc "styles" setting
# (core.styles). An org admin builds a full custom theme (Org settings → Slide
# styles): colours, fonts, typography sizes and table colours, seeded from one
# of the presets as a "Quick pick". It seeds the ``theme`` of every new deck
# authored in that org, so decks match the org's brand without the model having
# to choose. Stored on ``Organization.preferences["slide_theme"]``.
#
# Two stored shapes are supported:
#   * NEW (full custom): ``{"colors": {...}, "fonts": {...}, "typography": {...},
#     "tables": {...}}`` — what the settings form saves.
#   * OLD (preset only): ``{"name": <preset>}`` — kept for back-compat; expands
#     to that preset's fields.
# An org that never configured a style seeds nothing (decks inherit the base).
# ---------------------------------------------------------------------------
DEFAULT_SLIDE_THEME = "forest"

# Colour levers the org style exposes (each is a base-theme colour slot).
SLIDE_STYLE_COLOR_KEYS = (
    "lt1", "dk1", "dk2", "lt2",
    "accent1", "accent2", "accent3", "accent4", "accent5", "accent6",
    "success", "warning", "danger",
)

# Font roles the org can pick. Values MUST be a bundled family (a directory
# under ``core/assets/fonts``): the Pillow preview resolves a face by that
# directory name, so — unlike doc styles, which substitutes at export — a name
# outside the bundle would silently fall back to Carlito. SLIDE_FONT_FAMILIES is
# the allow-list; each is metric-compatible with a common proprietary face.
SLIDE_STYLE_FONT_KEYS = ("headline", "subhead", "body", "data")
SLIDE_FONT_FAMILIES = (
    ("Caladea", "Caladea — Cambria-style serif"),
    ("Tinos", "Tinos — Times-style serif"),
    ("Gelasio", "Gelasio — Georgia-style serif"),
    ("EBGaramond", "EB Garamond — serif"),
    ("Carlito", "Carlito — Calibri-style sans"),
    ("Arimo", "Arimo — Arial-style sans"),
    ("Cousine", "Cousine — Courier-style mono"),
)
_SLIDE_FONT_KEYS = frozenset(k for k, _ in SLIDE_FONT_FAMILIES)

# Typography sizes exposed: form key -> (text-style key, min pt, max pt).
SLIDE_STYLE_SIZE_KEYS = {
    "headline_size": ("headline", 18, 60),
    "subhead_size": ("subhead", 10, 40),
    "body_size": ("body", 8, 28),
}
SLIDE_BULLET_MAX = 4

# Table colour levers (each maps straight onto a ``theme["table"]`` key).
SLIDE_STYLE_TABLE_KEYS = ("header_fill", "header_color", "band_fill", "grid_color")

# Presentation metadata for the settings form (key, label, group). Colocated
# with the keys so the two never drift; consumed by the org-settings template.
SLIDE_COLOR_FIELD_META = (
    ("lt1", "Background", "Core"),
    ("dk1", "Body text", "Core"),
    ("dk2", "Headings", "Core"),
    ("lt2", "Band / soft fill", "Core"),
    ("accent1", "Accent 1 (primary)", "Accents"),
    ("accent2", "Accent 2", "Accents"),
    ("accent3", "Accent 3", "Accents"),
    ("accent4", "Accent 4", "Accents"),
    ("accent5", "Accent 5", "Accents"),
    ("accent6", "Accent 6", "Accents"),
    ("success", "Success", "Semantic"),
    ("warning", "Warning", "Semantic"),
    ("danger", "Danger", "Semantic"),
)
SLIDE_FONT_FIELD_META = (
    ("headline", "Headline / titles"),
    ("subhead", "Subheads"),
    ("body", "Body text"),
    ("data", "Data / tables"),
)
SLIDE_SIZE_FIELD_META = (
    ("headline_size", "Headline size", 18, 60),
    ("subhead_size", "Subhead size", 10, 40),
    ("body_size", "Body size", 8, 28),
)
SLIDE_TABLE_FIELD_META = (
    ("header_fill", "Header fill"),
    ("header_color", "Header text"),
    ("band_fill", "Banded row fill"),
    ("grid_color", "Gridlines"),
)


def _hex_hash(value) -> str:
    """A stored/derived colour as ``#RRGGBB`` (fallback ``#000000``)."""
    nh = _norm_hex(value) if isinstance(value, str) else None
    return "#" + nh if nh else "#000000"


def _slot_hex(colors: dict, ref: str) -> str:
    """Resolve a table colour ref (slot name or #hex) to a concrete ``#RRGGBB``."""
    if isinstance(ref, str) and ref.startswith("#"):
        return _hex_hash(ref)
    return _hex_hash(colors.get(ref))


def slide_style_defaults() -> dict:
    """The full slide-style field set at base-theme values (form defaults)."""
    return _full_style_from_override({})


def _full_style_from_override(override: dict) -> dict:
    """Resolve a sparse theme ``override`` to the full org-style field set.

    Colours and table colours come out as concrete ``#RRGGBB`` (table slots are
    dereferenced), fonts as bundled family keys, sizes as ints. Used to expand a
    preset (or the base) into form fields.
    """
    theme = _deep_merge(WILFRED_BASE_THEME, override or {})
    colors = theme["colors"]
    base_fonts = WILFRED_BASE_THEME["fonts"]
    return {
        "colors": {k: _hex_hash(colors.get(k)) for k in SLIDE_STYLE_COLOR_KEYS},
        "fonts": {
            k: (theme["fonts"].get(k) if theme["fonts"].get(k) in _SLIDE_FONT_KEYS else base_fonts[k])
            for k in SLIDE_STYLE_FONT_KEYS
        },
        "typography": {
            "headline_size": int(theme["text_styles"]["headline"]["size"]),
            "subhead_size": int(theme["text_styles"]["subhead"]["size"]),
            "body_size": int(theme["text_styles"]["body"]["size"]),
            "bullet_char": theme["bullet"]["char"],
        },
        "tables": {k: _slot_hex(colors, theme["table"][k]) for k in SLIDE_STYLE_TABLE_KEYS},
    }


def _is_full_style(stored) -> bool:
    """True when a stored value is the NEW full-custom shape (not ``{"name":…}``)."""
    return isinstance(stored, dict) and any(
        k in stored for k in ("colors", "fonts", "typography", "tables")
    )


def get_org_slide_style(org) -> dict:
    """Resolve an org's full slide-style field set (colours/fonts/typography/tables).

    ``org`` may be ``None`` (no membership) → base defaults. Tolerant of malformed
    stored values (falls back to defaults / the base preset). This is what the
    settings form binds to; :func:`org_slide_theme_override` turns it into a deck
    theme override.
    """
    stored = (getattr(org, "preferences", None) or {}).get("slide_theme") if org is not None else None
    if _is_full_style(stored):
        clean, err = validate_org_slide_style(stored)
        return clean if (clean and not err) else slide_style_defaults()
    # Old shape {"name": preset} or nothing — expand the preset into fields.
    name = stored.get("name") if isinstance(stored, dict) else None
    if name not in PRESET_THEMES:
        name = DEFAULT_SLIDE_THEME
    return _full_style_from_override(preset_theme_override(name))


def validate_org_slide_style(data) -> tuple[dict | None, str | None]:
    """Validate a slide-style payload from the org settings endpoint.

    Returns ``(clean, None)`` — ``clean`` is the full field set
    (``colors``/``fonts``/``typography``/``tables``) — or ``(None, error)``. A
    bare ``{"name": <preset>}`` is accepted and expanded (back-compat / a raw
    quick-pick save).
    """
    if not isinstance(data, dict):
        return None, "Invalid slide style payload."

    # Quick path: a bare preset name expands to that preset's fields.
    if not _is_full_style(data):
        name = data.get("name", DEFAULT_SLIDE_THEME)
        if not isinstance(name, str) or name not in PRESET_THEMES:
            return None, "Choose one of the available slide themes."
        return _full_style_from_override(preset_theme_override(name)), None

    defaults = slide_style_defaults()
    clean: dict = {"colors": {}, "fonts": {}, "typography": {}, "tables": {}}

    colors_in = data.get("colors") or {}
    if not isinstance(colors_in, dict):
        return None, "Invalid colours."
    for key in SLIDE_STYLE_COLOR_KEYS:
        val = colors_in.get(key, defaults["colors"][key])
        nh = _norm_hex(val) if isinstance(val, str) else None
        if not nh:
            return None, "Colours must be hex like #2563EB."
        clean["colors"][key] = "#" + nh

    fonts_in = data.get("fonts") or {}
    if not isinstance(fonts_in, dict):
        return None, "Invalid fonts."
    for key in SLIDE_STYLE_FONT_KEYS:
        val = fonts_in.get(key, defaults["fonts"][key])
        if val not in _SLIDE_FONT_KEYS:
            return None, "Pick a slide font from the list."
        clean["fonts"][key] = val

    typo_in = data.get("typography") or {}
    if not isinstance(typo_in, dict):
        return None, "Invalid typography."
    for size_key, (_style, lo, hi) in SLIDE_STYLE_SIZE_KEYS.items():
        val = typo_in.get(size_key, defaults["typography"][size_key])
        try:
            iv = int(val)
        except (TypeError, ValueError):
            return None, f"{size_key.replace('_', ' ').title()} must be a number."
        if not (lo <= iv <= hi):
            return None, f"{size_key.replace('_', ' ').title()} must be between {lo} and {hi}."
        clean["typography"][size_key] = iv
    bullet = typo_in.get("bullet_char", defaults["typography"]["bullet_char"])
    if not isinstance(bullet, str) or not bullet.strip():
        bullet = defaults["typography"]["bullet_char"]
    clean["typography"]["bullet_char"] = bullet.strip()[:SLIDE_BULLET_MAX]

    tables_in = data.get("tables") or {}
    if not isinstance(tables_in, dict):
        return None, "Invalid table colours."
    for key in SLIDE_STYLE_TABLE_KEYS:
        val = tables_in.get(key, defaults["tables"][key])
        nh = _norm_hex(val) if isinstance(val, str) else None
        if not nh:
            return None, "Table colours must be hex like #1F3D30."
        clean["tables"][key] = "#" + nh

    return clean, None


def _style_to_override(style: dict) -> dict:
    """Turn a full org-style field set into a sparse deck ``theme`` override."""
    return {
        "colors": dict(style["colors"]),
        "fonts": dict(style["fonts"]),
        "text_styles": {
            "headline": {"size": style["typography"]["headline_size"]},
            "subhead": {"size": style["typography"]["subhead_size"]},
            "body": {"size": style["typography"]["body_size"]},
        },
        "bullet": {"char": style["typography"]["bullet_char"]},
        "table": dict(style["tables"]),
    }


def org_slide_theme_override(org) -> dict:
    """The sparse deck ``theme`` override for an org's configured slide style.

    Empty dict when the org never configured a style (or there's no org) — callers
    seed a new deck's ``theme`` with this only when it's non-empty, so unconfigured
    orgs leave decks on the base theme.
    """
    stored = (getattr(org, "preferences", None) or {}).get("slide_theme") if org is not None else None
    if _is_full_style(stored):
        clean, err = validate_org_slide_style(stored)
        return _style_to_override(clean) if (clean and not err) else {}
    # Old shape {"name": preset} — resolve via the preset (forest → empty).
    name = stored.get("name") if isinstance(stored, dict) else None
    if name not in PRESET_THEMES:
        name = DEFAULT_SLIDE_THEME
    return preset_theme_override(name) or {}


def preset_style_fields() -> dict:
    """``{preset_name: full-field-set}`` — the quick-pick populators for the form."""
    return {name: _full_style_from_override(spec["theme"]) for name, spec in PRESET_THEMES.items()}


def org_slide_theme_swatch(org) -> dict:
    """``{bg, dark, accent}`` for the org theme's swatch in the in-deck picker."""
    colors = get_org_slide_style(org)["colors"]
    return {"bg": colors["lt1"], "dark": colors["dk2"], "accent": colors["accent1"]}


# ---------------------------------------------------------------------------
# User custom slide themes — a user can save their own palettes (from the slide
# panel's theme picker) and reuse them across decks. Stored on
# ``UserSettings.preferences["slide_themes"]`` as a list of entries:
#   {"id": "c<hex>", "label": str, "base": <preset>, "colors": {accent1,dk2,lt1}}
# A custom theme = a base preset with three brand colours overridden (accent,
# heading/dark, background); everything else inherits the base so decks stay
# coherent. Ids are namespaced ("c…") so they can't collide with preset names.
# ---------------------------------------------------------------------------
import re as _re  # noqa: E402 — local alias, keep the module header clean
import uuid as _uuid  # noqa: E402

MAX_USER_SLIDE_THEMES = 12
# The brand levers a custom theme exposes -> the native slot each maps to.
CUSTOM_THEME_COLOR_KEYS = ("accent1", "dk2", "lt1")
_CUSTOM_HEX_RE = _re.compile(r"^#[0-9A-Fa-f]{6}$")
_CUSTOM_LABEL_MAX = 40


def validate_custom_slide_theme(data) -> tuple[dict | None, str | None]:
    """Validate a user's custom-theme payload (label + base preset + 3 colours).

    Returns ``(clean, None)`` — clean has ``label``/``base``/``colors`` but no
    ``id`` (the caller assigns one) — or ``(None, error_message)``.
    """
    if not isinstance(data, dict):
        return None, "Invalid theme payload."
    label = data.get("label")
    if not isinstance(label, str) or not label.strip():
        return None, "Give the theme a name."
    label = label.strip()[:_CUSTOM_LABEL_MAX]

    base = data.get("base", DEFAULT_SLIDE_THEME)
    if not isinstance(base, str) or base not in PRESET_THEMES:
        return None, "Pick a base theme."

    colors_in = data.get("colors")
    if not isinstance(colors_in, dict):
        return None, "Choose the theme colours."
    colors = {}
    for key in CUSTOM_THEME_COLOR_KEYS:
        val = colors_in.get(key)
        if not isinstance(val, str) or not _CUSTOM_HEX_RE.match(val.strip()):
            return None, "Colours must be hex like #2563EB."
        colors[key] = val.strip().upper()

    return {"label": label, "base": base, "colors": colors}, None


def user_slide_themes(user) -> list[dict]:
    """The user's saved custom themes (sanitised list, never raises)."""
    prefs = _user_prefs(user)
    raw = prefs.get("slide_themes") if isinstance(prefs, dict) else None
    if not isinstance(raw, list):
        return []
    out = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        clean, err = validate_custom_slide_theme(entry)
        if err or not isinstance(entry.get("id"), str):
            continue
        clean["id"] = entry["id"]
        out.append(clean)
    return out


def _user_prefs(user) -> dict:
    """Read a user's stored preferences dict (best-effort, no raise)."""
    try:
        from accounts.models import UserSettings

        us = UserSettings.objects.filter(user=user).first()
        return (us.preferences or {}) if us else {}
    except Exception:  # noqa: BLE001
        return {}


def new_custom_theme_id() -> str:
    """A fresh id for a custom theme, namespaced so it can't be a preset name."""
    return "c" + _uuid.uuid4().hex[:10]


def custom_theme_override(entry: dict) -> dict:
    """The sparse deck ``theme`` override for a stored custom-theme entry."""
    base = preset_theme_override(entry.get("base")) or {}
    return _deep_merge(base, {"colors": dict(entry.get("colors") or {})})


def _swatch_from_override(override: dict) -> dict:
    """Resolve an override's bg/dark/accent hexes for a picker swatch."""
    colors = _deep_merge(WILFRED_BASE_THEME, override)["colors"]
    return {"bg": colors["lt1"], "dark": colors["dk2"], "accent": colors["accent1"]}


def user_slide_theme_swatches(user) -> list[dict]:
    """``[{id,label,bg,dark,accent}]`` for the user's custom themes (picker UI)."""
    out = []
    for entry in user_slide_themes(user):
        sw = _swatch_from_override(custom_theme_override(entry))
        out.append({"id": entry["id"], "label": entry["label"], **sw})
    return out


def resolve_named_slide_theme(name: str, user=None) -> dict | None:
    """Resolve a theme *name* to its sparse override — a preset, or (with a
    ``user``) one of that user's saved custom themes. ``None`` if unknown."""
    preset = preset_theme_override(name)
    if preset is not None:
        return preset
    if user is not None:
        for entry in user_slide_themes(user):
            if entry["id"] == name:
                return custom_theme_override(entry)
    return None


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


def _norm_hex(raw: str) -> str | None:
    """Normalise a hex colour to 6 upper-case digits, expanding 3-digit shorthand.
    Returns None for anything that isn't valid hex — so a bad literal degrades to
    'no colour' instead of crashing python-pptx's strict ``RGBColor.from_string``."""
    h = raw.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) == 6:
        try:
            int(h, 16)
        except ValueError:
            return None
        return h.upper()
    return None


def resolve_color(theme: dict, value: str | None) -> tuple[str, str] | None:
    """Resolve a colour value to ``("theme", slot)`` or ``("rgb", "RRGGBB")``.

    * ``None``/empty -> ``None`` (no colour set).
    * ``#RGB``/``#RRGGBB`` -> ``("rgb", "RRGGBB")`` (shorthand expanded, validated).
    * a native slot name -> ``("theme", slot)`` (the builder maps to
      MSO_THEME_COLOR so it tracks the template scheme + shows in the picker).
    * a semantic/other name present in ``theme["colors"]`` -> ``("rgb", hex)``.
    * anything unresolvable (incl. malformed hex) -> ``None``.
    """
    if not value:
        return None
    if value.startswith("#"):
        nh = _norm_hex(value)
        return ("rgb", nh) if nh else None
    if value in NATIVE_SLOTS:
        return ("theme", value)
    hexv = theme["colors"].get(value)
    if isinstance(hexv, str) and hexv.startswith("#"):
        nh = _norm_hex(hexv)
        return ("rgb", nh) if nh else None
    return None
