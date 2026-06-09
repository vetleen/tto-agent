"""Template filters for the usage screens."""
from django import template

register = template.Library()


@register.filter
def compact_number(value):
    """Render a count compactly: thousands get a separator, millions/billions abbreviate.

    3412 -> "3,412" · 9_842_150 -> "9.84M" · 1_233_890 -> "1.23M". The precise value
    is kept in a title= tooltip at the call site since abbreviation is lossy.
    """
    try:
        n = int(value)
    except (TypeError, ValueError):
        return value
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= 1_000_000_000:
        return f"{sign}{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{sign}{n / 1_000_000:.2f}M"
    return f"{sign}{n:,}"
