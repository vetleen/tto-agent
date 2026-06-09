"""Branding-related template helpers (Wilfred design system)."""
from django import template

register = template.Library()

# Avatar tone cycle — forest-light / forest / copper. A given seed (e.g. a
# user's email) always maps to the same tone, so an identity is stable in colour
# while different identities vary. (bg, fg) pairs use fixed brand hexes so they
# read on both light and dark message surfaces.
_AVATAR_TONES = (
    ("#DDEAE2", "#0B2418"),  # forest-100 on forest-900
    ("#16432C", "#F2F7F4"),  # forest-700 on near-white
    ("#BE8242", "#FFFFFF"),  # copper-500 on white
)


@register.filter
def avatar_style(seed):
    """Return an inline ``background``/``color`` style for an initials avatar.

    Usage: ``<span style="{{ user.email|avatar_style }}">{{ initial }}</span>``
    """
    s = str(seed or "")
    idx = sum(ord(ch) for ch in s) % len(_AVATAR_TONES)
    bg, fg = _AVATAR_TONES[idx]
    return f"background:{bg};color:{fg}"
