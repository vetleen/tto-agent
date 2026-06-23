"""Branding-related template helpers (Wilfred design system)."""
from django import template
from django.utils.html import format_html

register = template.Library()

# Per-variant (img classes, initials-span classes) for user_avatar. "nav" is the
# top-bar dropdown trigger; "chat" reuses the .wf-avatar rule from chat.html.
_AVATAR_VARIANTS = {
    "nav": (
        "inline-block h-8 w-8 shrink-0 rounded-full object-cover",
        "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-semibold",
    ),
    "chat": ("wf-avatar", "wf-avatar"),
}

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


@register.simple_tag
def user_avatar(user, variant="chat"):
    """Render a user's avatar: their uploaded picture if set, else an initials chip.

    Keeps the picture-or-initials branch in one place so the nav and chat stay in
    sync. ``variant`` selects the sizing/classes ("nav" or "chat").
    """
    img_class, span_class = _AVATAR_VARIANTS.get(variant, _AVATAR_VARIANTS["chat"])
    pic = getattr(user, "profile_picture", None)
    if pic:
        return format_html('<img src="{}" alt="" class="{}">', pic.url, img_class)
    email = getattr(user, "email", "") or ""
    return format_html(
        '<span class="{}" style="{}">{}</span>',
        span_class, avatar_style(email), email[:1].upper(),
    )


@register.filter
def compact_timesince(value):
    """Compact relative timestamp for the chat ledger sidebar.

    ``now`` under a minute, ``5m`` / ``3h`` within the day, a weekday
    abbreviation (``Tue``) within the last week, else ``Apr 30``.
    """
    if not value:
        return ""
    from django.utils import timezone

    now = timezone.now()
    try:
        delta = now - value
    except TypeError:
        return ""
    secs = delta.total_seconds()
    if secs < 60:
        return "now"
    if secs < 3600:
        return f"{int(secs // 60)}m"
    if secs < 86400:
        return f"{int(secs // 3600)}h"
    local = timezone.localtime(value) if timezone.is_aware(value) else value
    if delta.days < 7:
        return local.strftime("%a")
    # Portable across platforms (avoid %-d which fails on Windows).
    return f"{local.strftime('%b')} {local.day}"
