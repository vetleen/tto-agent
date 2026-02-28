"""Template filters for documents app."""
from django.template import Library
from django.template.defaultfilters import truncatechars

register = Library()


@register.filter
def truncate_project_name(value, max_chars=60):
    """Truncate project name for display. Optional max_chars (default 60). Adds '...' if truncated."""
    if value is None:
        return ""
    try:
        n = int(max_chars)
    except (TypeError, ValueError):
        n = 60
    return truncatechars(value, n)
